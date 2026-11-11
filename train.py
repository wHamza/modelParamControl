import random
import subprocess
import argparse
import numpy as np
import torch
import torch.nn as nn
import wandb
import yaml
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_git(*args):
    result = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def get_git_hash():
    git_hash = run_git("rev-parse", "--short", "HEAD")
    # uncommitted changes mean the hash does not describe the code that actually ran
    if run_git("status", "--porcelain"):
        git_hash += "-dirty"
    return git_hash


class CIFAR10HF(Dataset):
    def __init__(self, split, transform):
        ds = load_dataset("uoft-cs/cifar10", split=split)
        # decode PNGs once and keep them in RAM; decoding on every step slows training
        self.data = np.stack([np.array(img) for img in ds["img"]])
        self.labels = ds["label"]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.transform(self.data[i]), self.labels[i]


def get_loaders(cfg):
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(MEAN, STD),
    ])

    # dedicated generator: shuffle order stays the same even if other code draws random numbers
    generator = torch.Generator().manual_seed(cfg["seed"])

    train_loader = DataLoader(
        CIFAR10HF("train", transform), batch_size=cfg["batch_size"], shuffle=True,
        generator=generator, num_workers=2, pin_memory=True,
    )
    test_loader = DataLoader(
        CIFAR10HF("test", transform), batch_size=cfg["batch_size"], shuffle=False,
        num_workers=2, pin_memory=True,
    )
    return train_loader, test_loader


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        # no bias: the BatchNorm right after has its own shift
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    def __init__(self, width, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, width),
            conv_block(width, 2 * width),
            conv_block(2 * width, 4 * width),
        )
        self.head = nn.Linear(4 * width, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.mean(dim=(2, 3))
        return self.head(x)


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    running_loss = 0.0
    grad_norm_sum = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()

        gradNorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))

        optimizer.step()
        # OneCycleLR is defined per batch, not per epoch
        scheduler.step()

        # weight by batch size so the last, smaller batch does not skew the mean
        running_loss += loss.item() * inputs.size(0)
        grad_norm_sum += gradNorm.item()
    return running_loss / len(loader.dataset), grad_norm_sum / len(loader)
    
  


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct = 0.0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        running_loss += criterion(outputs, targets).item() * inputs.size(0)
        correct += (outputs.argmax(dim=1) == targets).sum().item()
    n = len(loader.dataset)
    return running_loss / n, correct / n

def log_predictions(model, loader, device, n = 16):
    model.eval()
    images = []
    CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                    "dog", "frog", "horse", "ship", "truck"]
    inputs, targets = next(iter(loader))
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    for i in range(n):
        with torch.no_grad():
            preds = model(inputs.to(device)).argmax(dim=1).cpu()
            img = (inputs[i] * std + mean).clamp(0, 1)
            img = (img.permute(1, 2, 0) * 255).byte().numpy()
            images.append(wandb.Image(img, caption=f"pred: {CLASSES[preds[i]]} / true: {CLASSES[targets[i]]}"))

    wandb.log({"predictions": images})
            


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--group", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config("config.yml")
    # CLI seed overrides the config so seed runs need no file edits
    if args.seed is not None:
        cfg["seed"] = args.seed
    set_seed(cfg["seed"])
    cfg = load_config("config.yml")
    # before the model is built: initial weights are random too
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run = wandb.init(project="exp-tracking", config=cfg, group=args.group)
    cfg = dict(run.config)
    run.config.update({
        "git_hash": get_git_hash(),
        "torch_version": torch.__version__,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "none",
    })

    train_loader, test_loader = get_loaders(cfg)
    model = SmallCNN(cfg["width"]).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=cfg["lr"], momentum=0.9, weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg["lr"], epochs=cfg["epoch"], steps_per_epoch=len(train_loader),
    )

    for epoch in range(cfg["epoch"]):
        train_loss, grad_norm= train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        lr = scheduler.get_last_lr()[0]

        print(f"epoch {epoch + 1} | train {train_loss:.3f} | test {test_loss:.3f} "
              f"| acc {test_acc:.3f} | lr {lr:.4f}")
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "test/loss": test_loss,
            "test/acc": test_acc,
            "lr": lr,
            "train/grad_norm":grad_norm, 
        })
        log_predictions(model, test_loader, device, 16)
    

    run.finish()


if __name__ == "__main__":
    main()
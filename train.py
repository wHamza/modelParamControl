import yaml
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import torch.nn as nn
import numpy as np
import random
import subprocess
g = torch.Generator()
g.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(b):
    np.random.seed(b)
    random.seed(b)
    torch.manual_seed(b)
    print(random.random(), np.random.rand(), torch.randn(1))
result = subprocess.run(
    ["git", "rev-parse", "--short","HEAD"],
    #["git", "status" ,"--porcelain"],
    
    capture_output=True, text=True, check=True,
)
control = subprocess.run(
    ["git", "status" ,"--porcelain"],
    capture_output=True, text=True, check= True,
)
def get_git_hash():
    hash = result.stdout.strip()
    if control.stdout.strip():
        hash += "-dirty"
    return hash
def get_loaders(cfg):

    MEAN = (0.4914, 0.4822, 0.4465)
    STD = (0.2470, 0.2435, 0.2616)

    
    with open(cfg, 'r') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    
    train_tf = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(MEAN, STD),
        ])
    train_ds = datasets.CIFAR10("data", train = True,download=True, transform=train_tf)
    test_ds = datasets.CIFAR10("data", train = False, download = True, transform = train_tf)

    train_loader = DataLoader(train_ds, batch_size=data['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, generator=g,batch_size=data['batch_size'], shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, test_loader, data


def conv_block(in_ch, out_ch):
    return nn.Sequential(
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
            conv_block(width, 2*width),
            conv_block(2* width, 4*width),
        )
        self.head = nn.Linear(4*width, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.mean(dim=(2, 3))
        return self.head(x)

def one_epoch_loop(model, loader, optimizer, schedular, criterion, device):
    model.train()
    running_loss = 0
    for inputs, targets in loader:
        inputs = inputs.to(device) 
        targets = targets.to(device)

        optimizer.zero_grad()
        
        outs = model(inputs)
        loss =criterion(outs, targets)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        schedular.step()
    return running_loss
    
def evaluate(model, loader, criterion):
    with torch.no_grad():
        model.eval()
        correct = 0
        running_loss = 0
        for ins, targets in loader:
            ins = ins.to(device)
            targets = targets.to(device)

            outs = model(ins)
            loss = criterion(outs, targets)

            correct += (outs.argmax(dim=1) == targets).sum().item()

            running_loss += loss.item() * ins.size(0)
    return running_loss / len(loader.dataset), correct / len(loader.dataset)






def main():
    train_loader, test_loader, data =get_loaders()
    model = SmallCNN(32)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),momentum=0.9 , lr = data['lr'], weight_decay=data['weight_decay'])
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, epochs=5, max_lr=data['lr'], steps_per_epoch = 391)

    for epoch in range(data["epoch"]):
        train_loss = one_epoch_loop(model, train_loader, optimizer, criterion, device, schedular=lr_scheduler)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(train_loss, test_loss, test_acc)



if __name__ == "__main__":
    main()
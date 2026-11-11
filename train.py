import yaml
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import numpy as np
import random
import subprocess
g = torch.Generator()
g.manual_seed(42)

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
    with open(cfg, 'r') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    
    train_tf = v2.Compose([
            v2.ToTensor(),
            v2.Normalize(),
        ])
    train_ds = datasets.CIFAR10("data", train = True,download=True, transform=train_tf)
    test_ds = datasets.CIFAR10("data", train = False, download = True, transform = train_tf)

    train_loader = DataLoader(train_ds, batch_size=data['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, generator=g,batch_size=data['batch_size'], shuffle=False, num_workers=2, pin_memory=True)


    print(next(iter(train_loader)))




    



def main():
    get_loaders('config.yml')


if __name__ == "__main__":
    main()
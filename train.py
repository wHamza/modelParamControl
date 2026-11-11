import yaml
import torch
import numpy as np
import random
import subprocess
    
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

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(40)
    with open('config.yml', 'r') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    print(data, "\n", torch.randn(1), device, "\n", get_git_hash(), "\n", torch.__version__)


if __name__ == "__main__":
    main()
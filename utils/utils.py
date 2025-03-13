import torch
import subprocess
from . import binvox_rw

def visualize_from_tensor(tensor:torch.tensor):
    pass

def visualize_from_file(path_to_file:str):
    cmd = [
        "/home/mahmoud-sayed/Desktop/Graduation Project/current/visualizer/viewvox",
        path_to_file
    ]
    subprocess.run(cmd)

def binarize_tensor(tensor:torch.tensor):
    pass

def save_tensor_as_bvox(path_to_dir,tensor:torch.tensor):
    pass





def load_binvox_as_tensor(path:str):
    with open(path, 'rb') as f:
        model = binvox_rw.read_as_3d_array(f)
    tensor = torch.tensor(model.data)

    return tensor

import torch


def get_torch_device():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    return device

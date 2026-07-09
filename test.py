import math
import torch
import torch.nn
import torch.optim
import torchvision
import numpy as np
from model import *
import config as c
import datasets
import modules.Unet_common as common
from modules.dwt1d import DWT1D_3Level, IWT1D_3Level
import soundfile as sf

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load(path):
    state = torch.load(path, map_location=device)
    sd = {k: v for k, v in state['net'].items() if 'tmp_var' not in k}

    # 依照目前 net 是否 DataParallel，自動調整 key
    is_dp = isinstance(net, torch.nn.DataParallel)
    has_module = next(iter(sd)).startswith("module.")

    if is_dp and (not has_module):
        sd = {"module." + k: v for k, v in sd.items()}
    if (not is_dp) and has_module:
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

    net.load_state_dict(sd, strict=True)

    if 'opt' in state:
        try:
            optim.load_state_dict(state['opt'])
        except:
            pass

def gauss_noise_like(x):
    return torch.randn_like(x)



def computePSNR(origin,pred):
    origin = np.array(origin)
    origin = origin.astype(np.float32)
    pred = np.array(pred)
    pred = pred.astype(np.float32)
    mse = np.mean((origin/1.0 - pred/1.0) ** 2 )
    if mse < 1.0e-10:
      return 100
    return 10 * math.log10(255.0**2/mse)

def _to_tensor(x):
    # x: Tensor / numpy / list-of-Tensor / list-of-numpy
    if torch.is_tensor(x):
        return x

    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)

    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            raise ValueError("Empty list in batch")

        # list of Tensor
        if torch.is_tensor(x[0]):
            return torch.stack(x, dim=0)

        # list of numpy
        if isinstance(x[0], np.ndarray):
            return torch.stack([torch.from_numpy(t) for t in x], dim=0)

        # list of numbers
        return torch.tensor(x)

    raise TypeError(f"Unsupported type: {type(x)}")

def to_device_batch(batch, device):
    # batch can be (cover, secret) or dict
    if isinstance(batch, (list, tuple)):
        cover, secret = batch
    elif isinstance(batch, dict):
        cover, secret = batch["cover"], batch["secret"]
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    cover = _to_tensor(cover).to(device)
    secret = _to_tensor(secret).to(device)
    return cover, secret

net = Model()
net.cpu()
init_model(net)
net = torch.nn.DataParallel(net, device_ids=c.device_ids)
params_trainable = (list(filter(lambda p: p.requires_grad, net.parameters())))
optim = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, c.weight_step, gamma=c.gamma)

load(c.MODEL_PATH + c.suffix)

net.eval()

# 改用 3 次連續斬與還原
dwt = DWT1D_3Level()
iwt = IWT1D_3Level()


with torch.no_grad():
    for i, batch in enumerate(datasets.testloader):
        cover, secret = to_device_batch(batch, device)   # (B,C,L)

        cover_d = dwt(cover)      # (B,2C,L/2)
        secret_d = dwt(secret)    # (B,2C,L/2)
        x = torch.cat([cover_d, secret_d], dim=1)        # (B,4C,L/2)

        y = net(x, rev=False)
        y_steg = y.narrow(1, 0, 8 * c.channels_in)       # (B,2C,L/2)
        y_z    = y.narrow(1, 8 * c.channels_in, y.shape[1] - 8 * c.channels_in)

        steg = iwt(y_steg)        # (B,C,L)

        z = gauss_noise_like(y_z)
        x_hat = net(torch.cat([y_steg, z], dim=1), rev=True)
        secret_hat_d = x_hat.narrow(1, 8 * c.channels_in, x_hat.shape[1] - 8 * c.channels_in)
        secret_rev = iwt(secret_hat_d)

        print("cover", cover.shape, "secret", secret.shape, "steg", steg.shape, "secret_rev", secret_rev.shape)

        # 存第一筆就好
        sf.write("cover.wav", cover[0,0].cpu().numpy(), c.host_sr)
        sf.write("secret.wav", secret[0,0].cpu().numpy(), c.host_sr)
        sf.write("steg.wav", steg[0,0].cpu().numpy(), c.host_sr)
        sf.write("secret_rev.wav", secret_rev[0,0].cpu().numpy(), c.host_sr)
        break





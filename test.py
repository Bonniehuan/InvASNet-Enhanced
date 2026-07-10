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

# 🎯 這裡換成了我們專屬的聲音 SNR 測量器
def calculate_snr(original_audio, processed_audio):
    orig = original_audio.cpu().float()
    proc = processed_audio.cpu().float()
    signal_power = torch.sum(orig ** 2)
    noise_power = torch.sum((orig - proc) ** 2)
    if noise_power < 1e-10:
        return 100.0
    return (10 * torch.log10(signal_power / noise_power)).item()

def _to_tensor(x):
    if torch.is_tensor(x): return x
    if isinstance(x, np.ndarray): return torch.from_numpy(x)
    if isinstance(x, (list, tuple)):
        if len(x) == 0: raise ValueError("Empty list in batch")
        if torch.is_tensor(x[0]): return torch.stack(x, dim=0)
        if isinstance(x[0], np.ndarray): return torch.stack([torch.from_numpy(t) for t in x], dim=0)
        return torch.tensor(x)
    raise TypeError(f"Unsupported type: {type(x)}")

def to_device_batch(batch, device):
    if isinstance(batch, (list, tuple)):
        cover, secret = batch
    elif isinstance(batch, dict):
        cover, secret = batch["cover"], batch["secret"]
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")
    return _to_tensor(cover).to(device), _to_tensor(secret).to(device)

net = Model()
net.cpu()
init_model(net)
net = torch.nn.DataParallel(net, device_ids=c.device_ids)
params_trainable = (list(filter(lambda p: p.requires_grad, net.parameters())))
optim = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)

# 讀取 config.py 設定好的 model.pt 滿級武器
load(c.MODEL_PATH + c.suffix)
net.eval()

# 召喚你改裝好的 3-Level 頻率大水管
dwt = DWT1D_3Level()
iwt = IWT1D_3Level()

with torch.no_grad():
    for i, batch in enumerate(datasets.testloader):
        cover, secret = to_device_batch(batch, device)

        # 1. 拆解與偽裝
        cover_d = dwt(cover)      
        secret_d = dwt(secret)    
        x = torch.cat([cover_d, secret_d], dim=1)        

        y = net(x, rev=False)
        y_steg = y.narrow(1, 0, 8 * c.channels_in)       
        y_z    = y.narrow(1, 8 * c.channels_in, y.shape[1] - 8 * c.channels_in)

        steg = iwt(y_steg)        

        # 2. 🎯 零雜訊輔助還原 (防炸耳雷包拆除)
        z = torch.zeros_like(y_z) 
        
        x_hat = net(torch.cat([y_steg, z], dim=1), rev=True)
        secret_hat_d = x_hat.narrow(1, 8 * c.channels_in, x_hat.shape[1] - 8 * c.channels_in)
        secret_rev = iwt(secret_hat_d)

        # 3. 🎯 終極數值防爆夾
        steg = torch.clamp(steg, min=-1.0, max=1.0)
        secret_rev = torch.clamp(secret_rev, min=-1.0, max=1.0)

        # 4. 🎯 結算成績單 (保證呼叫名稱一致，絕不報錯！)
        steg_snr = calculate_snr(cover, steg)
        rev_snr = calculate_snr(secret, secret_rev)
        
        print("\n" + "="*50)
        print("🎉 恭喜破關！完美產出音檔！")
        print(f"🎵 偽裝音樂 (Stego) SNR: {steg_snr:.2f} dB")
        print(f"🗣️ 還原語音 (Secret) SNR: {rev_snr:.2f} dB")
        print("="*50 + "\n")

        # 存檔見真章
        sf.write("cover.wav", cover[0,0].cpu().numpy(), c.host_sr)
        sf.write("secret.wav", secret[0,0].cpu().numpy(), c.host_sr)
        sf.write("steg.wav", steg[0,0].cpu().numpy(), c.host_sr)
        sf.write("secret_rev.wav", secret_rev[0,0].cpu().numpy(), c.host_sr)
        break

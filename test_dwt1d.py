import torch
from modules.dwt1d import DWT1D_3Level, IWT1D_3Level

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    B, C, L = 4, 1, 44160  # 論文 segment length
    x = torch.randn(B, C, L, device=device)

    dwt = DWT1D_3Level().to(device)
    iwt = IWT1D_3Level().to(device)

    y = dwt(x)
    x_hat = iwt(y)

    max_err = (x - x_hat).abs().max().item()
    mean_err = (x - x_hat).abs().mean().item()

    print("x      :", tuple(x.shape))
    print("dwt(x) :", tuple(y.shape))
    print("iwt(dwt(x)):", tuple(x_hat.shape))
    print("max_abs_error :", max_err)
    print("mean_abs_error:", mean_err)

if __name__ == "__main__":
    main()


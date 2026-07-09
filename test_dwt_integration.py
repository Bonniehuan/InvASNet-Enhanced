import torch
from modules.dwt1d import DWT1D_3Level, IWT1D_3Level
from invblock import INV_block

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    B, C, L = 4, 1, 44160

    cover = torch.randn(B, C, L, device=device).clamp(-1, 1)
    secret = torch.randn(B, C, L, device=device).clamp(-1, 1)

    dwt = DWT1D_3Level().to(device)
    iwt = IWT1D_3Level().to(device)
    blk = INV_block(in_1=1, in_2=1, harr=True).to(device)

    cover_dwt = dwt(cover)     # (B,2,L/2)
    secret_dwt = dwt(secret)   # (B,2,L/2)

    x = torch.cat([cover_dwt, secret_dwt], dim=1)  # (B,4,L/2)

    y = blk(x, rev=False)      # (B,4,L/2)
    y_cover = y[:, :2, :]      # (B,2,L/2)  -> stego
    stego = iwt(y_cover)       # (B,1,L)

    # backward (use the rest as z, or just use the same split for now)
    # simplest: feed y back with same y (sanity). In full paper, z is replaced by noise.
    x_hat = blk(y, rev=True)   # (B,4,L/2)
    cover_hat = iwt(x_hat[:, :2, :])
    secret_hat = iwt(x_hat[:, 2:, :])

    print("cover      :", tuple(cover.shape))
    print("secret     :", tuple(secret.shape))
    print("stego      :", tuple(stego.shape))
    print("cover_hat  :", tuple(cover_hat.shape))
    print("secret_hat :", tuple(secret_hat.shape))

    print("cover max abs err :", (cover - cover_hat).abs().max().item())
    print("secret max abs err:", (secret - secret_hat).abs().max().item())

if __name__ == "__main__":
    main()


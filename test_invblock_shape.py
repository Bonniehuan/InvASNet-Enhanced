import torch
import config as c
from invblock import INV_block

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    B = 4
    C = 1
    L = 44160

    # 1D Haar 後通道變 2C
    # 你的 invblock 期待輸入是 cat(cover_dwt, secret_dwt)
    # cover_dwt: (B,2C,L/2), secret_dwt: (B,2C,L/2) => concat => (B,4C,L/2)
    x = torch.randn(B, 4*C, L//2, device=device)

    blk = INV_block(in_1=C, in_2=C, harr=True).to(device)

    y = blk(x, rev=False)
    x_hat = blk(y, rev=True)

    print("x      :", tuple(x.shape))
    print("y      :", tuple(y.shape))
    print("x_hat  :", tuple(x_hat.shape))
    print("max_abs_error:", (x - x_hat).abs().max().item())

if __name__ == "__main__":
    main()


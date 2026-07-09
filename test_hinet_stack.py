import torch
from hinet import Hinet

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    B, L2 = 4, 22080  # L/2
    x = torch.randn(B, 4, L2, device=device)

    net = Hinet().to(device)

    y = net(x, rev=False)
    x_hat = net(y, rev=True)

    print("x      :", tuple(x.shape))
    print("y      :", tuple(y.shape))
    print("x_hat  :", tuple(x_hat.shape))
    print("max_abs_error:", (x - x_hat).abs().max().item())

if __name__ == "__main__":
    main()


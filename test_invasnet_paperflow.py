import torch
import config as c
from hinet import Hinet
from modules.dwt1d import DWT1D_3Level, IWT1D_3Level

def gauss_noise_like(x):
    return torch.randn_like(x)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    B = 4
    C = c.channels_in
    L = c.segment_length

    # fake audio (later will be real wav)
    cover  = torch.randn(B, C, L, device=device).clamp(-1, 1)
    secret = torch.randn(B, C, L, device=device).clamp(-1, 1)

    dwt = DWT1D_3Level().to(device)
    iwt = IWT1D_3Level().to(device)
    net = Hinet().to(device)
    net.eval()

    # -------------------------
    # Forward Concealing (paper)
    # -------------------------
    cover_dwt  = dwt(cover)    # (B, 2C, L/2)
    secret_dwt = dwt(secret)   # (B, 2C, L/2)

    x = torch.cat([cover_dwt, secret_dwt], dim=1)  # (B, 4C, L/2)

    with torch.no_grad():
        y = net(x, rev=False)  # (B, 4C, L/2)

    stego_dwt = y[:, :2*C, :]          # (B, 2C, L/2)
    z_true    = y[:, 2*C:4*C, :]       # (B, 2C, L/2)  -> lost info in paper

    stego = iwt(stego_dwt)             # (B, C, L)

    # -------------------------
    # Backward Revealing (paper)
    # Replace z_true with random z
    # -------------------------
    z_rand = gauss_noise_like(z_true)

    y_rev = torch.cat([stego_dwt, z_rand], dim=1)  # (B, 4C, L/2)

    with torch.no_grad():
        x_hat = net(y_rev, rev=True)   # (B, 4C, L/2)

    cover_hat_dwt  = x_hat[:, :2*C, :]
    secret_hat_dwt = x_hat[:, 2*C:4*C, :]

    cover_hat  = iwt(cover_hat_dwt)
    secret_hat = iwt(secret_hat_dwt)

    # -------------------------
    # Print sanity metrics
    # -------------------------
    print("cover      :", tuple(cover.shape))
    print("secret     :", tuple(secret.shape))
    print("stego      :", tuple(stego.shape))
    print("cover_hat  :", tuple(cover_hat.shape))
    print("secret_hat :", tuple(secret_hat.shape))

    # In paper flow, secret_hat won't match secret BEFORE training (random weights),
    # but shapes must be correct and no crash.
    print("stego- cover max abs:", (stego - cover).abs().max().item())
    print("secret_hat - secret max abs:", (secret_hat - secret).abs().max().item())

if __name__ == "__main__":
    main()


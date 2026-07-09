import torch
import torch.nn as nn

# =========================================
# 裝備 1：原本的 1 次小波 (你提供的原始碼)
# =========================================
class DWT1D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"DWT1D expects (B,C,L), got {tuple(x.shape)}")
        B, C, L = x.shape
        if L % 2 != 0:


            raise ValueError(f"Length L must be even for Haar DWT, got L={L}")

        x_even = x[:, :, 0::2]   
        x_odd  = x[:, :, 1::2]   

        low  = (x_even + x_odd) * 0.5   
        high = (x_even - x_odd) * 0.5   

        return torch.cat([low, high], dim=1)

class IWT1D(nn.Module):
    """
    Inverse 1D Haar DWT.
    Input:  y shape (B, 2C, L/2)  where first C channels are low, next C are high
    Output: x shape (B, C, L)

    x_even = low + high
    x_odd  = low - high
    """
    def __init__(self):
        super().__init__()

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        if y.dim() != 3:
            raise ValueError(f"IWT1D expects (B,2C,L/2), got {tuple(y.shape)}")
        B, CC, Lh = y.shape
        if CC % 2 != 0:
            raise ValueError(f"Channel dim must be even (2C), got {CC}")

        C = CC // 2
        low  = y[:, :C, :]
        high = y[:, C:, :]

        x_even = low + high
        x_odd  = low - high

        # interleave even/odd back to length 2*Lh
        L = Lh * 2
        x = torch.empty((B, C, L), device=y.device, dtype=y.dtype)
        x[:, :, 0::2] = x_even
        x[:, :, 1::2] = x_odd
        return x

# =========================================
# 裝備 2：我們剛寫好的 3 次小波擴充包
# =========================================
class DWT1D_3Level(nn.Module):
    def __init__(self):
        super().__init__()
        self.dwt = DWT1D() 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.dwt(x)    # 第 1 斬
        x2 = self.dwt(x1)   # 第 2 斬
        x3 = self.dwt(x2)   # 第 3 斬
        return x3

class IWT1D_3Level(nn.Module):
    def __init__(self):
        super().__init__()
        self.iwt = IWT1D()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.iwt(x)  # 解開第 3 刀
        x = self.iwt(x)  # 解開第 2 刀
        x = self.iwt(x)  # 解開第 1 刀
        return x

# =========================================
# 測試板機：在本地 CPU 驗證水管有沒有接對
# =========================================
if __name__ == "__main__":
    # 1. 隨機捏一個假的 1 秒鐘單聲道音訊 (Batch=1, Channel=1, Length=16000)
    dummy_input = torch.randn(1, 1, 16000)
    print("Input shape:", dummy_input.shape)

    # 2. 把我們的新武器拿出來
    my_weapon = DWT1D_3Level()

    # 3. 扣板機！把假音訊丟進去切 3 次
    output = my_weapon(dummy_input)

    # 4. 驗收成果！
    print("Output shape:", output.shape)

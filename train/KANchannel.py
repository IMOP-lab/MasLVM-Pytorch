import torch
import torch.nn as nn
import numpy as np

# 定义 KANLayer 类
class KANLayer(nn.Module):
    def __init__(self, in_dim=3, out_dim=2, num=5, k=3, noise_scale=0.1, scale_base=1.0, scale_sp=1.0, base_fun=torch.nn.SiLU(), grid_eps=0.02, grid_range=[-1, 1], sp_trainable=True, sb_trainable=True, device='cuda'):
        super(KANLayer, self).__init__()
        self.size = size = out_dim * in_dim
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.num = num
        self.k = k
        self.device = device

        self.grid = torch.einsum('i,j->ij', torch.ones(size, device=device), torch.linspace(grid_range[0], grid_range[1], steps=num + 1, device=device))
        self.grid = torch.nn.Parameter(self.grid).requires_grad_(False)
        noises = (torch.rand(size, self.grid.shape[1], device=device) - 1 / 2) * noise_scale / num
        self.coef = torch.nn.Parameter(self.curve2coef(self.grid, noises, self.grid, k, device))
        if isinstance(scale_base, float):
            self.scale_base = torch.nn.Parameter(torch.ones(size, device=device) * scale_base).requires_grad_(sb_trainable)
        else:
            self.scale_base = torch.nn.Parameter(torch.FloatTensor(scale_base).to(device)).requires_grad_(sb_trainable)
        self.scale_sp = torch.nn.Parameter(torch.ones(size, device=device) * scale_sp).requires_grad_(sp_trainable)
        self.base_fun = base_fun

        self.mask = torch.nn.Parameter(torch.ones(size, device=device)).requires_grad_(False)
        self.grid_eps = grid_eps
        self.weight_sharing = torch.arange(size, device=device)
        self.lock_counter = 0
        self.lock_id = torch.zeros(size, device=device)

    def forward(self, x):
        batch, channels = x.shape
        
        # 将 x 扩展为 (batch, out_dim * in_dim)
        x = torch.einsum('ij,k->ikj', x, torch.ones(self.out_dim, device=self.device)).reshape(batch, self.size).permute(1, 0)
        
        preacts = x.permute(1, 0).clone().reshape(batch, self.out_dim, self.in_dim)
        base = self.base_fun(x).permute(1, 0)
        
        # 使用 KAN 层计算输出
        weight_sharing = self.weight_sharing.to(self.grid.device)  # 确保索引和张量在同一设备
        y = self.coef2curve(x_eval=x, grid=self.grid[weight_sharing], coef=self.coef[weight_sharing], k=self.k, device=self.device)
        y = y.permute(1, 0)
        postspline = y.clone().reshape(batch, self.out_dim, self.in_dim)
        
        y = self.scale_base.unsqueeze(dim=0) * base + self.scale_sp.unsqueeze(dim=0) * y
        y = self.mask[None, :] * y
        postacts = y.clone().reshape(batch, self.out_dim, self.in_dim)
        
        # 将 y 变回 (batch, out_dim)
        y = torch.sum(y.reshape(batch, self.out_dim, self.in_dim), dim=2)
        
        return y, preacts, postacts, postspline

    def update_grid_from_samples(self, x):
        batch = x.shape[0]
        x = torch.einsum('ij,k->ikj', x, torch.ones(self.out_dim, ).to(self.device)).reshape(batch, self.size).permute(1, 0)
        x_pos = torch.sort(x, dim=1)[0]
        y_eval = self.coef2curve(x_pos, self.grid, self.coef, self.k, device=self.device)
        num_interval = self.grid.shape[1] - 1
        ids = [int(batch / num_interval * i) for i in range(num_interval)] + [-1]
        grid_adaptive = x_pos[:, ids]
        margin = 0.01
        grid_uniform = torch.cat([grid_adaptive[:, [0]] - margin + (grid_adaptive[:, [-1]] - grid_adaptive[:, [0]] + 2 * margin) * a for a in np.linspace(0, 1, num=self.grid.shape[1])], dim=1)
        self.grid.data = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        self.coef.data = self.curve2coef(x_pos, y_eval, self.grid, self.k, device=self.device)

    def initialize_grid_from_parent(self, parent, x):
        batch = x.shape[0]
        x_eval = torch.einsum('ij,k->ikj', x, torch.ones(self.out_dim, ).to(self.device)).reshape(batch, self.size).permute(1, 0)
        x_pos = parent.grid
        sp2 = KANLayer(in_dim=1, out_dim=self.size, k=1, num=x_pos.shape[1] - 1, scale_base=0., device=self.device)
        sp2.coef.data = self.curve2coef(sp2.grid, x_pos, sp2.grid, k=1, device=self.device)
        y_eval = self.coef2curve(x_eval, parent.grid, parent.coef, parent.k, device=self.device)
        percentile = torch.linspace(-1, 1, self.num + 1).to(self.device)
        self.grid.data = sp2(percentile.unsqueeze(dim=1))[0].permute(1, 0)
        self.coef.data = self.curve2coef(x_eval, y_eval, self.grid, self.k, self.device)

    def get_subset(self, in_id, out_id):
        spb = KANLayer(len(in_id), len(out_id), self.num, self.k, base_fun=self.base_fun, device=self.device)
        spb.grid.data = self.grid.reshape(self.out_dim, self.in_dim, spb.num + 1)[out_id][:, in_id].reshape(-1, spb.num + 1)
        spb.coef.data = self.coef.reshape(self.out_dim, self.in_dim, spb.coef.shape[1])[out_id][:, in_id].reshape(-1, spb.coef.shape[1])
        spb.scale_base.data = self.scale_base.reshape(self.out_dim, self.in_dim)[out_id][:, in_id].reshape(-1, )
        spb.scale_sp.data = self.scale_sp.reshape(self.out_dim, self.in_dim)[out_id][:, in_id].reshape(-1, )
        spb.mask.data = self.mask.reshape(self.out_dim, self.in_dim)[out_id][:, in_id].reshape(-1, )
        spb.in_dim = len(in_id)
        spb.out_dim = len(out_id)
        spb.size = spb.in_dim * spb.out_dim
        return spb

    def lock(self, ids):
        self.lock_counter += 1
        for i in range(len(ids)):
            if i != 0:
                self.weight_sharing[ids[i][1] * self.in_dim + ids[i][0]] = ids[0][1] * self.in_dim + ids[0][0]
            self.lock_id[ids[i][1] * self.in_dim + ids[i][0]] = self.lock_counter

    def unlock(self, ids):
        num = len(ids)
        locked = True
        for i in range(num):
            locked *= (self.weight_sharing[ids[i][1] * self.in_dim + ids[i][0]] == self.weight_sharing[ids[0][1] * self.in_dim + ids[0][0]])
        if locked == False:
            print("they are not locked. unlock failed.")
            return 0
        for i in range(len(ids)):
            self.weight_sharing[ids[i][1] * self.in_dim + ids[i][0]] = ids[i][1] * self.in_dim + ids[i][0]
            self.lock_id[ids[i][1] * self.in_dim + ids[i][0]] = 0
        self.lock_counter -= 1

    @staticmethod
    def B_batch(x, grid, k=0, extend=True, device='cuda'):
        def extend_grid(grid, k_extend=0):
            h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)
            for i in range(k_extend):
                grid = torch.cat([grid[:, [0]] - h, grid], dim=1)
                grid = torch.cat([grid, grid[:, [-1]] + h], dim=1)
            grid = grid.to(device)
            return grid

        if extend:
            grid = extend_grid(grid, k_extend=k)

        grid = grid.unsqueeze(dim=2).to(device)
        x = x.unsqueeze(dim=1).to(device)

        if k == 0:
            value = (x >= grid[:, :-1]) * (x < grid[:, 1:])
        else:
            B_km1 = KANLayer.B_batch(x[:, 0], grid=grid[:, :, 0], k=k - 1, extend=False, device=device)
            value = (x - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)]) * B_km1[:, :-1] + (
                        grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * B_km1[:, 1:]
        return value

    @staticmethod
    def coef2curve(x_eval, grid, coef, k, device="cuda"):
        if coef.dtype != x_eval.dtype:
            coef = coef.to(x_eval.dtype)
        y_eval = torch.einsum('ij,ijk->ik', coef, KANLayer.B_batch(x_eval, grid, k, device=device))
        return y_eval

    @staticmethod
    def curve2coef(x_eval, y_eval, grid, k, device="cuda"):
        mat = KANLayer.B_batch(x_eval, grid, k, device=device).permute(0, 2, 1)
        coef = torch.linalg.lstsq(mat.to(device), y_eval.unsqueeze(dim=2).to(device), driver='gels').solution[:, :, 0]
        return coef.to(device)


# 定义 KAN 通道注意力机制
class KANChannelAttention(nn.Module):
    def __init__(self, in_channels, grid=3, k=3):
        super(KANChannelAttention, self).__init__()
        self.in_channels = in_channels
        self.kan_layer = KANLayer(in_dim=in_channels, out_dim=in_channels, num=grid, k=k)
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # 计算每个通道的全局平均池化 (batch_size, in_channels, 1, 1)
        avg_out = torch.mean(x, dim=(2, 3), keepdim=False)
        
        # 使用 KANLayer 计算注意力权重 (batch_size, in_channels)
        attention, _, _, _ = self.kan_layer(avg_out)
        
        # 使用 Sigmoid 激活函数将权重归一化到 [0, 1]
        attention = torch.sigmoid(attention)
        
        # 将注意力权重扩展回原始特征图的维度 (batch_size, in_channels, 1, 1)
        attention = attention.unsqueeze(-1).unsqueeze(-1)
        
        # 将注意力权重应用到原始特征图上
        x = x * attention.expand_as(x)
        
        return x

# 测试 KAN 通道注意力机制
if __name__ == "__main__":
    input_tensor = torch.randn(8, 64, 32, 32)  # 假设输入是 8 个 64 通道的 32x32 特征图
    kan_attention = KANChannelAttention(in_channels=64)
    output = kan_attention(input_tensor)
    print("Output Shape:", output.shape)  # 输出的形状应与输入形状相同

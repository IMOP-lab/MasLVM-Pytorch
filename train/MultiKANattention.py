import torch
import torch.nn as nn
from KANself import KANSelfAttention
import torch.nn.functional as F

class KANMultiEncoderAttention(nn.Module):
    def __init__(self, in_channels, num_encoders=3, grid=3, k=3, chunk_size=64):
        super(KANMultiEncoderAttention, self).__init__()
        self.num_encoders = num_encoders
        self.in_channels = in_channels
        self.chunk_size = chunk_size  # 设置块大小

        # 为每个编码器的输出定义一个自注意力机制
        self.attention_layers = nn.ModuleList([
            KANSelfAttention(in_channels, grid, k) for _ in range(num_encoders)
        ])

        # 定义一个输出卷积层，将多编码器的输出合并
        self.output_conv = nn.Conv2d(in_channels * num_encoders, in_channels, kernel_size=1, bias=False)

    def forward(self, *encoder_outputs):
        assert len(encoder_outputs) == self.num_encoders, "输入的编码器数量不匹配"

        # 获取输入的空间维度
        batch_size, channels, height, width = encoder_outputs[0].size()

        # 对每个编码器的输出进行逐块处理
        attention_outputs = []
        for i, attention_layer in enumerate(self.attention_layers):
            encoder_output = encoder_outputs[i]

            # 将特征图拆分为多个块
            unfolded = F.unfold(encoder_output, kernel_size=self.chunk_size, stride=self.chunk_size)
            unfolded = unfolded.view(batch_size, channels, self.chunk_size, self.chunk_size, -1).permute(0, 4, 1, 2, 3)  # (batch_size, num_chunks, channels, chunk_size, chunk_size)
            num_chunks = unfolded.size(1)
            attention_chunks = []

            for j in range(num_chunks):
                chunk = unfolded[:, j]
                att_chunk = attention_layer(chunk)  # 应用自注意力机制到每个块
                attention_chunks.append(att_chunk.unsqueeze(1))  # 保持维度

            # 合并所有块
            attention_chunks = torch.cat(attention_chunks, dim=1)
            attention_chunks = attention_chunks.permute(0, 2, 3, 4, 1).reshape(batch_size, channels * self.chunk_size * self.chunk_size, -1)
            folded = F.fold(attention_chunks, output_size=(height, width), kernel_size=self.chunk_size, stride=self.chunk_size)

            attention_outputs.append(folded)

        # 将多个注意力输出在通道维度上进行连接
        concatenated_output = torch.cat(attention_outputs, dim=1)

        # 使用卷积层将连接的输出变换回原始的通道数
        final_output = self.output_conv(concatenated_output)

        return final_output

# 测试 KAN 多编码器注意力机制
if __name__ == "__main__":
    # 假设我们有三个编码器，每个编码器的输出都是 64 通道的 32x32 特征图
    encoder_output1 = torch.randn(8, 64, 32, 32)
    encoder_output2 = torch.randn(8, 64, 32, 32)
    encoder_output3 = torch.randn(8, 64, 32, 32)
    
    # 创建多编码器注意力机制
    kan_multi_attention = KANMultiEncoderAttention(in_channels=64, num_encoders=3)
    
    # 将三个编码器的输出传递给注意力机制
    output = kan_multi_attention(encoder_output1, encoder_output2, encoder_output3)
    
    print("Output Shape:", output.shape)  # 输出的形状应与每个输入的编码器输出形状相同 (batch_size, in_channels, H, W)

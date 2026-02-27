      
"""
=== self attention处理Token序列的完整流程 ===
核心原理：将Token序列之间两两做点积得到scores，在得到相同维度的token序列
输入维度（batch_size, token_len, token_dim) 
输出维度(batch_size, token_len, token_dim)

转换路径：
1. 输入Token序列：首先有一个输入的Token序列，每个Token会被表示成一个d_model维的向量，输入张量形状为 [batch_size, seq_len, d_model]，其中batch_size是批次大小，seq_len是序列长度，d_model是模型的特征维度。
2. 生成Q、K、V矩阵：将输入的Token序列分别通过三个线性变换层W_Q、W_K、W_V，得到查询矩阵Q、键矩阵K和值矩阵V。这三个矩阵的形状均为 [batch_size, seq_len, d_k * n_heads]（这里假设多头注意力的头数为n_heads，每个头的维度为d_k）。
3. 多头分割：将Q、K、V矩阵按照头的数量进行分割，调整形状为 [batch_size, n_heads, seq_len, d_k]，以便在每个头中独立计算注意力。
4. 计算注意力分数：在每个头中，通过计算查询矩阵Q和键矩阵K的转置的点积，并除以缩放因子sqrt(d_k)，得到注意力分数矩阵scores，形状为 [batch_size, n_heads, seq_len, seq_len]。
5. 应用掩码：如果存在填充掩码pad_mask（形状为 [batch_size, seq_len, seq_len]），将其扩展到每个头（形状变为 [batch_size, n_heads, seq_len, seq_len]），并将掩码位置对应的scores置为负无穷，这样在后续的softmax操作中这些位置的概率就会趋近于0。
6. 计算注意力权重：对scores矩阵进行softmax操作，得到注意力权重矩阵attn，形状同样为 [batch_size, n_heads, seq_len, seq_len]。
7. 加权求和：使用注意力权重矩阵attn对值矩阵V进行加权求和，得到上下文矩阵context，形状为 [batch_size, n_heads, seq_len, d_v]（d_v通常和d_k相等）。
8. 多头合并：将每个头的上下文矩阵合并，调整形状为 [batch_size, seq_len, n_heads * d_v]。
9. 线性变换：通过一个线性变换层将合并后的矩阵映射回原始的特征维度d_model，得到多头注意力的输出，形状为 [batch_size, seq_len, d_model]。
10. 前馈网络：多头注意力的输出经过一个前馈神经网络（包含两个线性变换层和一个激活函数），得到最终的输出，形状仍然为 [batch_size, seq_len, d_model]。

在整个过程中，还使用了残差连接和层归一化来提高模型的训练稳定性和性能。
"""

import torch
from torch import nn
import torch
from math import sqrt as msqrt

# 定义一个配置类，用于存储模型的超参数
class Config:
    def __init__(self):
        # 序列最大长度
        self.max_len = 30
        # 最大词汇量
        self.max_vocab = 50
        # 最大预测数量
        self.max_pred = 5
        # 多头注意力中每个头的查询和键的维度
        self.d_k = 64
        # 多头注意力中每个头的值的维度
        self.d_v = 64
        # 模型的特征维度
        self.d_model = 768
        # 前馈神经网络的中间层维度
        self.d_ff = self.d_model * 4
        # 多头注意力的头数
        self.n_heads = 12
        # 编码器层数
        self.n_layers = 6
        # 句子段的数量
        self.n_segs = 2
        # Dropout概率
        self.p_dropout = 0.1
        # BERT掩码概率
        self.p_mask = 0.8
        # BERT替换概率
        self.p_replace = 0.1
        # BERT不做处理的概率
        self.p_do_nothing = 1 - self.p_mask - self.p_replace
        # 设备选择
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device)
        # 批次大小
        self.batch_size = 6
        # 训练轮数
        self.epochs = 30
        # 学习率
        self.lr = 1e-3


# GELU激活函数
def gelu(x):
    '''
    GELU激活函数的两种实现方式：
    1. 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    2. 0.5 * x * (1. + torch.erf(torch.sqrt(x, 2)))
    '''
    return 0.5 * x * (1. + torch.erf(x / msqrt(2.)))

# 缩放点积注意力机制
class ScaledDotProductAttention(nn.Module):
    def __init__(self, config):
        super(ScaledDotProductAttention, self).__init__()
        self.config = config

    def forward(self, Q, K, V, attn_mask):
        '''
        前向传播函数
        参数：
        Q: 查询矩阵，形状为[batch, n_heads, seq_len, d_k]
        K: 键矩阵，形状为[batch, n_heads, seq_len, d_k]
        V: 值矩阵，形状为[batch, n_heads, seq_len, d_v]
        attn_mask: 注意力掩码，形状为[batch, seq_len, seq_len]
        返回值：
        context: 注意力加权后的上下文向量，形状为[batch, n_heads, seq_len, d_v]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / msqrt(self.config.d_k)  # 计算点积并缩放
        scores.masked_fill_(attn_mask, -1e9)  # 应用掩码
        attn = nn.Softmax(dim=-1)(scores)  # 应用Softmax函数
        context = torch.matmul(attn, V)  # 计算上下文向量
        return context


# 多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        self.config = config
        # 查询矩阵的线性变换
        self.W_Q = nn.Linear(self.config.d_model, self.config.d_k * self.config.n_heads, bias=False)
        # 键矩阵的线性变换
        self.W_K = nn.Linear(self.config.d_model, self.config.d_k * self.config.n_heads, bias=False)
        # 值矩阵的线性变换
        self.W_V = nn.Linear(self.config.d_model, self.config.d_v * self.config.n_heads, bias=False)
        # 输出的线性变换
        self.fc = nn.Linear(self.config.n_heads * self.config.d_v, self.config.d_model, bias=False)

    def forward(self, Q, K, V, attn_mask):
        '''
        前向传播函数
        参数：
        Q: 查询矩阵，形状为[batch, seq_len, d_model]
        K: 键矩阵，形状为[batch, seq_len, d_model]
        V: 值矩阵，形状为[batch, seq_len, d_model]
        attn_mask: 注意力掩码，形状为[batch, seq_len, seq_len]
        返回值：
        output: 多头注意力的输出，形状为[batch, seq_len, d_model]
        '''
        batch = Q.size(0)
        per_Q = self.W_Q(Q).view(batch, -1, self.config.n_heads, self.config.d_k).transpose(1, 2)  # 分头处理查询矩阵
        per_K = self.W_K(K).view(batch, -1, self.config.n_heads, self.config.d_k).transpose(1, 2)  # 分头处理键矩阵
        per_V = self.W_V(V).view(batch, -1, self.config.n_heads, self.config.d_v).transpose(1, 2)  # 分头处理值矩阵
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.config.n_heads, 1, 1)  # 扩展注意力掩码
        context = ScaledDotProductAttention(self.config)(per_Q, per_K, per_V, attn_mask)  # 应用缩放点积注意力机制
        context = context.transpose(1, 2).contiguous().view(batch, -1, self.config.n_heads * self.config.d_v)  # 合并头
        output = self.fc(context)  # 应用输出的线性变换
        return output


# 前馈神经网络
class FeedForwardNetwork(nn.Module):
    def __init__(self, config):
        super(FeedForwardNetwork, self).__init__()
        # 第一层线性变换
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        # 第二层线性变换
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        # Dropout层
        self.dropout = nn.Dropout(config.p_dropout)
        # GELU激活函数
        self.gelu = gelu

    def forward(self, x):
        '''
        前向传播函数
        参数：
        x: 输入的序列，形状为[batch, seq_len, d_model]
        返回值：
        输出的序列，形状为[batch, seq_len, d_model]
        '''
        x = self.fc1(x)  # 第一层线性变换
        x = self.dropout(x)  # 应用Dropout
        x = self.gelu(x)  # 应用GELU激活函数
        x = self.fc2(x)  # 第二层线性变换
        return x


# 编码器层
class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        # 第一个归一化层
        self.norm1 = nn.LayerNorm(config.d_model)
        # 第二个归一化层
        self.norm2 = nn.LayerNorm(config.d_model)
        # 多头注意力机制
        self.enc_attn = MultiHeadAttention(config)
        # 前馈神经网络
        self.ffn = FeedForwardNetwork(config)

    def forward(self, x, pad_mask):
        '''
        前向传播函数
        参数：
        x: 输入的序列，形状为[batch, seq_len, d_model]
        pad_mask: 掩码矩阵，形状为[batch, seq_len, seq_len]
        返回值：
        输出的序列，形状为[batch, seq_len, d_model]
        '''
        residual = x  # 保存残差连接
        x = self.norm1(x)  # 第一个归一化层
        x = self.enc_attn(x, x, x, pad_mask) + residual  # 应用多头注意力机制并加上残差连接
        residual = x  # 保存残差连接
        x = self.norm2(x)  # 第二个归一化层
        x = self.ffn(x) + residual  # 应用前馈神经网络并加上残差连接
        return x


if __name__ == "__main__":
    config = Config()
    # 创建编码器层实例
    encoder_layer = EncoderLayer(config)

    # 生成随机输入数据
    batch_size = 2
    seq_len = 10
    input_tensor = torch.randn(batch_size, seq_len, config.d_model)

    # 生成合理的填充掩码
    # 假设第一个序列长度为 6，第二个序列长度为 8
    seq_lengths = [6, 8]
    pad_mask = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.bool)
    for i, length in enumerate(seq_lengths):
        pad_mask[i, length:, :] = True
        pad_mask[i, :, length:] = True
    print("Input shape:", input_tensor.shape)
    # 进行前向传播
    output = encoder_layer(input_tensor, pad_mask)
    print("Output shape:", output.shape)

    
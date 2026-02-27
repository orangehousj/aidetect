      

import torch
import torch.nn as nn
import torch

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


# 嵌入层
class Embeddings(nn.Module):
    def __init__(self, config):
        super(Embeddings, self).__init__()
        # 句子段嵌入
        self.seg_emb = nn.Embedding(config.n_segs, config.d_model)
        # 单词嵌入
        self.word_emb = nn.Embedding(config.max_vocab, config.d_model)
        # 位置嵌入
        self.pos_emb = nn.Embedding(config.max_len, config.d_model)
        # 归一化层
        self.norm = nn.LayerNorm(config.d_model)
        # Dropout层
        self.dropout = nn.Dropout(config.p_dropout)
        self.config = config

    def forward(self, x, seg):
        '''
        前向传播函数
        参数：
        x: 输入的序列，形状为[batch, seq_len]
        seg: 句子段信息，形状为[batch, seq_len]
        返回值：
        嵌入后的序列，形状为[batch, seq_len, d_model]
        '''
        word_enc = self.word_emb(x)  # 单词嵌入
        pos = torch.arange(x.shape[1], dtype=torch.long, device=self.config.device)
        pos = pos.unsqueeze(0).expand_as(x)
        pos_enc = self.pos_emb(pos)  # 位置嵌入
        seg_enc = self.seg_emb(seg)  # 句子段嵌入
        x = self.norm(word_enc + pos_enc + seg_enc)  # 将三种嵌入相加并归一化
        return x



if __name__ == "__main__":
    # 初始化配置对象
    config = Config()

    # 创建 Embeddings 类的实例
    embeddings = Embeddings(config)
    embeddings = embeddings.to(config.device)
    # 创建一个简单的输入序列
    # 假设我们有一个 batch_size=2 的输入，每个序列长度为 10
    input_tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]], dtype=torch.long).to(config.device)

    # 创建句子段信息
    # 假设第一个句子的段 ID 为 0，第二个句子的段 ID 为 1
    segment_ids = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]], dtype=torch.long).to(config.device)

    # 通过嵌入层处理输入
    embedded_output = embeddings(input_tokens, segment_ids)

    # 打印输出结果
    print("嵌入后的输出形状:", embedded_output.shape)
    print("嵌入后的输出示例:", embedded_output[0, :5, :5])  # 打印前5个单词的前5个维度的嵌入结果

    
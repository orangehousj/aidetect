      
"""
=== 文本到Token的完整转换流程 ===
核心原理：将自然语言转化为计算机可理解的数字形式
转换路径：原始文本 → 清洗分词 → 构建词表 → 数字映射 → 嵌入表示

输入维度（batch_size, token_len) 
输出维度(batch_size, token_len, token_dim)
"""

import re
import torch
from torch import nn

def manual_tokenization(raw_text):
    """
    手动实现文本转Token（分步教学版）
    适用于教学场景，帮助理解底层原理
    """
    
    # ===== 阶段1：文本预处理 =====
    # 目标：将原始文本转化为规范化的单词序列
    # 原理：统一文本格式，消除无关干扰因素
    cleaned_text = re.sub(r"[.,!?\-]", '', raw_text.lower())  # 正则表达式去标点+转小写
    sentences = cleaned_text.split('\n')  # 按换行符分割独立句子
    # import pdb;pdb.set_trace()
    # 示例：原始文本清洗过程
    print("\n[预处理示例]")
    print("原始文本：", raw_text.split('\n')[0])
    print("清洗结果：", sentences[0])

    # ===== 阶段2：构建词汇表 =====
    # 目标：创建单词与数字ID的映射关系
    # 原理：建立语言元素的数字化字典
    special_tokens = {
        '[PAD]': 0,  # 填充占位符（用于统一序列长度）
        '[CLS]': 1,   # 分类标记（表示文本开始）
        '[SEP]': 2,   # 分隔标记（区分不同句子）
        '[MASK]': 3   # 掩码标记（用于训练时随机遮盖）
    }
    
    # 提取语料库全部唯一单词
    all_words = " ".join(sentences).split()
    unique_words = list(set(all_words))
    # ['won', 'shopping', 'to', 'congratulations', 'visit', 'great', 'too', 'thank', 'romeo', 'team', 'oh', 'she', 'about', 'how', 'competition', 'very', 'is', 'juliet', 'hello', 'you', 'meet', 'are', 'name', 'the', 'what', 'going', 'not', 'where', 'am', 'my', 'nice', 'well', 'today', 'i', 'grandmother', 'baseball']

    # 合并特殊标记与普通单词
    word2id = {**special_tokens, 
              **{word: idx+len(special_tokens) for idx, word in enumerate(unique_words)}}
    '''
    {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3, 'won': 4, 'shopping': 5, 'to': 6, 'congratulations': 7, 'visit': 8, 'great': 9, 'too': 10, 'thank': 11, 'romeo': 12, 'team': 13, 'oh': 14, 'she': 15, 'about': 16, 'how': 17, 'competition': 18, 'very': 19, 'is': 20, 'juliet': 21, 'hello': 22, 'you': 23, 'meet': 24, 'are': 25, 'name': 26, 'the': 27, 'what': 28, 'going': 29, 'not': 30, 'where': 31, 'am': 32, 'my': 33, 'nice': 34, 'well': 35, 'today': 36, 'i': 37, 'grandmother': 38, 'baseball': 39}
    '''
    # 验证映射完整性
    assert len(word2id) == len(special_tokens) + len(unique_words), "词表构建错误！"

    # ===== 阶段3：Token转换 =====
    # 目标：将文本序列转换为数字ID序列
    # 原理：通过字典查询实现单词到数字的映射
    tokenized = []
    for sent in sentences:
        # 拆分句子为单词列表
        words = sent.split()
        # 查询每个单词对应的ID
        ids = [word2id[word] for word in words]
        tokenized.append(ids)
    # [[22, 17, 25, 23, 37, 32, 12], [22, 12, 33, 26, 20, 21, 34, 6, 24, 23], [34, 24, 23, 10, 17, 25, 23, 36], [9, 33, 39, 13, 4, 27, 18], [14, 7, 21], [11, 23, 12], [31, 25, 23, 29, 36], [37, 32, 29, 5, 28, 16, 23], [37, 32, 29, 6, 8, 33, 38, 15, 20, 30, 19, 35]]
    # 转换过程示例
    print("\n[Token转换示例]")
    print("句子：", sentences[0])
    print("Token IDs：", tokenized[0])
    print("词汇表大小：", len(word2id))

def auto_tokenization(raw_text):
    """
    使用HuggingFace Transformers库实现（生产推荐）
    特点：使用预训练模型的分词器，包含子词切分等高级特性
    """
    from transformers import BertTokenizer

    # 初始化BERT分词器（自动下载预训练词表）
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # 参数设置
    max_length = 20  # 控制序列最大长度
    
    # 自动化处理流程
    encoded = tokenizer(
        raw_text,
        max_length=max_length,
        padding='max_length',  # 自动填充到指定长度
        truncation=True,       # 超长自动截断
        return_tensors="pt"    # 返回PyTorch张量
    )
    
    # 输出解析
    print("\n[自动化Tokenization]")
    print("输入文本：", raw_text.split('\n')[0])
    print("Token IDs：", encoded['input_ids'][0])
    print("注意力掩码：", encoded['attention_mask'][0])  # 1表示有效token
    print("分段标记：", encoded['token_type_ids'][0])    # 区分不同句子

def embedding_demo():
    """
    Token到向量的转换演示
    原理：通过嵌入层将离散ID映射为连续向量
    """
    # 参数设置
    vocab_size = 10000  # 词汇表容量
    embed_dim = 128     # 向量维度
    
    # 创建嵌入层：实质是一个可学习的查找表
    embedding_layer = nn.Embedding(vocab_size, embed_dim)
    # import pdb;pdb.set_trace()
    # 示例：将单个ID转换为向量
    sample_id = torch.tensor([17])
    vector = embedding_layer(sample_id)
    
    print("\n[嵌入表示示例]")
    print("原始ID：", sample_id.item())
    print("对应向量：", vector.shape)  # 输出形状：torch.Size([1, 128])

if __name__ == "__main__":
    # 莎士比亚风格对话数据集
    dialogue = (
        'Hello, how are you? I am Romeo.\n'       
        'Hello, Romeo My name is Juliet. Nice to meet you.\n'  
        'Nice meet you too. How are you today?\n'  
        'Great. My baseball team won the competition.\n'  
        'Oh Congratulations, Juliet\n'            
        'Thank you Romeo\n'                        
        'Where are you going today?\n'            
        'I am going shopping. What about you?\n'  
        'I am going to visit my grandmother. she is not very well'  
    )
    
    # 执行手动转换流程
    manual_tokenization(dialogue)
    
    # 执行自动转换流程
    auto_tokenization(dialogue)
    
    # 嵌入层演示
    embedding_demo()

    
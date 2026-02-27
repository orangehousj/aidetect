# 目录结构
.
├── dataset/             # 数据集目录，包含原始数据及划分后的训练/验证/测试集
├── demo/                # bert示例与演示代码，用于学习模型结构
├── hf_models/           # HuggingFace 本地模型缓存或离线模型文件
├── download_model.py    # 下载并缓存预训练模型到本地的脚本
├── fintune_bert.py      # BERT 微调主脚本：数据加载、训练、评估与模型保存
├── readme.md            # 项目说明文档，包含使用方法与运行步骤
├── requirements.txt     # 项目依赖库列表，用于环境配置
└── split_data.py        # 数据预处理与数据集划分脚本

# 到文件夹目录下
cd /root/project1/project3
# 安装环境
pip install -r requirements.txt
# 配置代理
source /etc/network_turbo
# 下载模型
python download_model.py 
# 微调
python fintune_bert.py
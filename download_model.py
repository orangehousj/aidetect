from transformers import BertTokenizer, BertModel

load_pretrain_path = "./hf_models/bert-base-uncased"

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", cache_dir=load_pretrain_path)
model = BertModel.from_pretrained("bert-base-uncased", cache_dir=load_pretrain_path)

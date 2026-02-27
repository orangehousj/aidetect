import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification

# -------------------- Dataset --------------------
class MyDataset(Dataset):
    """
    自定义数据集：
    - 从CSV读取数据
    - 文本转小写
    - 使用BERT tokenizer编码
    """
    def __init__(self, file_path, tokenizer, max_len=128):
        df = pd.read_csv(file_path)
        df['Cleaned_Text'] = df['answer'].fillna("").astype(str).str.lower()
        self.labels = torch.tensor(df['is_cheating'].astype(int).values, dtype=torch.long)

        self.encodings = tokenizer(
            df['Cleaned_Text'].tolist(),
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }

# -------------------- Main --------------------
def main():
    # ===== 设备 =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== 数据路径 =====
    train_file_path = '/root/project1/project3/dataset/train_data.csv'
    val_file_path   = '/root/project1/project3/dataset/val_data.csv'
    test_file_path  = '/root/project1/project3/dataset/test_data.csv'
    load_pretrain_path = "/root/project1/project3/hf_models/bert-base-uncased"
    max_len = 128
    batch_size = 64

    # ===== 训练设置 =====
    epochs = 10
    best_val_acc = 0.0
    save_dir = "/root/project1/project3/best_model"
    # 可以更换为 5e-5试试
    learning_rate = 5e-6

    # ===== 分词器 =====
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased",  cache_dir=load_pretrain_path, local_files_only=True)

    # ===== 数据集与加载器 =====
    train_dataset = MyDataset(train_file_path, tokenizer, max_len=max_len)
    val_dataset   = MyDataset(val_file_path, tokenizer, max_len=max_len)
    test_dataset  = MyDataset(test_file_path, tokenizer, max_len=max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size)

    # ===== 模型 =====
    # model = BertForSequenceClassification.from_pretrained(
    #     'bert-base-uncased', num_labels=2
    # ).to(device)
    model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",cache_dir=load_pretrain_path,
            local_files_only=True, num_labels=2
    ).to(device)

    # ===== 损失函数 =====
    loss_fn = torch.nn.CrossEntropyLoss()

    # ===== 优化器 =====
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # ===== 训练 + 验证 =====
    for epoch in range(epochs):
        # ---------- Train ----------
        model.train()
        train_loss_sum = 0.0
        train_correct, train_total = 0, 0

        for batch in train_loader:
            optimizer.zero_grad()

            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device)
            )
            logits = outputs.logits
            labels = batch['labels'].to(device)

            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_loss = train_loss_sum / len(train_loader)
        train_acc = train_correct / train_total

        # ---------- Validation ----------
        model.eval()
        val_loss_sum = 0.0
        val_correct, val_total = 0, 0

        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device)
                )
                logits = outputs.logits
                labels = batch['labels'].to(device)

                loss = loss_fn(logits, labels)
                val_loss_sum += loss.item()

                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss_sum / len(val_loader)
        val_acc = val_correct / val_total

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(save_dir)

    # ===== 测试（使用最佳模型） =====
    print("\nEvaluating best model on test set...")
    best_model = BertForSequenceClassification.from_pretrained(save_dir).to(device)
    best_model.eval()

    test_correct, test_total = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            outputs = best_model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device)
            )
            preds = torch.argmax(outputs.logits, dim=1)
            labels = batch['labels'].to(device)

            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)

    test_acc = test_correct / test_total
    print(f"Final Test Top1-Acc: {test_acc:.4f}")

if __name__ == "__main__":
    main()

import random
import pandas as pd


# 定义文件路径
all_data_path = "/root/project1/project3/dataset/all_data.csv"
train_path = "/root/project1/project3/dataset/train_data.csv"
val_path = "/root/project1/project3/dataset/val_data.csv"
test_path = "/root/project1/project3/dataset/test_data.csv "
# 设置随机种子，保证每次运行结果一致（同时设置python原生和pandas的种子）
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
pd.set_option('mode.chained_assignment', None)  # 关闭pandas的警告提示
pd.DataFrame.sample.random_state = RANDOM_SEED  # 设置pandas采样的随机种子

# 第一步：使用pandas读取CSV文件
try:
    df = pd.read_csv(all_data_path, encoding='utf-8')
except FileNotFoundError:
    raise FileNotFoundError(f"文件 {all_data_path} 不存在，请检查路径是否正确 ")
except Exception as e:
    raise Exception(f"读取文件失败：{str(e)}")


# 检查数据是否为空
if df.empty:
    raise ValueError("csv文件中没有有效数据")
total_data = len(df)
print(f" 原始数据总行数（含表头）：{total_data} (表头1行，数据行{total_data-1}行)")

# 第二步：随机打散数据行（设置random_state保证可复现）
df_shuffled = df.sample(frac=1,random_state=RANDOM_SEED)  # frac=1 表示采样全部数据
# 第三步：按7:1.5:1.5比例计算划分行数
train_ratio = 0.7
val_ratio = 0.15

train_num = int(total_data * train_ratio)
val_num = int(total_data * val_ratio)
test_num = total_data - train_num - val_num  # 补全剩余行数，避免小数误差

# 第四步：划分数据
train_df = df_shuffled.iloc[:train_num]
val_df = df_shuffled.iloc[train_num:train_num + val_num]
test_df = df_shuffled.iloc[train_num + val_num:]

# 第五步：写入划分后的文件（index=False 不写入行索引，header=True 保留表头）
try:
    train_df.to_csv(train_path, index=False, header=True, encoding='utf-8')
    val_df.to_csv(val_path, index=False, header=True, encoding='utf-8')
    test_df.to_csv(test_path, index=False, header=True, encoding='utf-8')
except Exception as e:
    raise Exception(f"写入文件失败：{str(e)}")

# 打印划分结果，方便验证
print(f"随机种子设置为：{RANDOM_SEED}")
print(f"训练集行数：{len(train_df)} (占比： {len(train_df)/total_data:.2f})")
print(f"验证集行数：{len(val_df)} (占比： {len(val_df)/total_data:.2f})")
print(f"测试集行数：{len(test_df)} (占比： {len(test_df)/total_data:.2f})")
print("数据已通过Pandas随机打散并按比例划分完成！")



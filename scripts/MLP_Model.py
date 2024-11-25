import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import WeightedRandomSampler

# 加载特征表格
data_path = "data/processed/final_features.csv"
final_features = pd.read_csv(data_path)

# 分离特征与目标变量
X = final_features.drop(columns=["AnswerId", "MisconceptionId"])  # 输入特征
y = final_features["MisconceptionId"]  # 输出目标

# 映射目标变量到从 0 开始的连续整数
unique_classes = sorted(y.unique())
class_mapping = {original: i for i, original in enumerate(unique_classes)}
inverse_class_mapping = {i: original for original, i in class_mapping.items()}
y_mapped = y.map(class_mapping)

# 对特征进行降维 (PCA) 和标准化
pca = PCA(n_components=200)  # 增加 PCA 维度
scaler = StandardScaler()
X_pca = pca.fit_transform(X)
X_scaled = scaler.fit_transform(X_pca)

# 检查类别分布
class_counts = y_mapped.value_counts()

# 筛选样本数 >= 2 的类别
valid_classes = class_counts[class_counts >= 2].index
class_mapping_filtered = {old: new for new, old in enumerate(valid_classes)}

# 过滤特征和目标变量，并重新映射类别
filtered_indices = y_mapped.isin(valid_classes)
X_filtered = X_scaled[filtered_indices]
y_filtered = y_mapped[filtered_indices].map(class_mapping_filtered)

# 重新划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_filtered, y_filtered, test_size=0.3, random_state=42, stratify=y_filtered
)

# 验证目标变量范围
assert y_train.min() >= 0 and y_train.max() < len(valid_classes), "Target labels out of range!"
assert y_test.min() >= 0 and y_test.max() < len(valid_classes), "Target labels out of range!"

# 重新计算权重并生成采样器
class_counts_filtered = y_train.value_counts()
class_weights_filtered = 1.0 / class_counts_filtered
weights_filtered = y_train.map(class_weights_filtered).values
sampler = WeightedRandomSampler(weights_filtered, len(weights_filtered))

# 转换为 PyTorch 的 Dataset 和 DataLoader
class Dataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels.values, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

train_dataset = Dataset(X_train, y_train)
test_dataset = Dataset(X_test, y_test)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=64, sampler=sampler, shuffle=False
)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# 定义改进的 MLP 模型
class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.fc4(x)
        return x

# 初始化模型
input_size = X_scaled.shape[1]
num_classes = len(valid_classes)  # 使用过滤后的类别数
model = MLP(input_size, num_classes)

# 重新计算权重并生成采样器
class_counts_filtered = y_train.value_counts()
class_weights_filtered = 1.0 / class_counts_filtered
class_weights_filtered = class_weights_filtered.reindex(valid_classes).fillna(0)  # 确保索引匹配

# 创建类别权重张量
class_weights_tensor = torch.tensor(
    [class_weights_filtered[unique_class] for unique_class in valid_classes],
    dtype=torch.float32
).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 检查是否有 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 训练模型
epochs = 30
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")

# 测试模型
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for features, labels in test_loader:
        features, labels = features.to(device), labels.to(device)
        outputs = model(features)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 评估模型
all_preds = pd.Series(all_preds).map(inverse_class_mapping)
all_labels = pd.Series(all_labels).map(inverse_class_mapping)

accuracy = accuracy_score(all_labels, all_preds)
macro_f1 = f1_score(all_labels, all_preds, average="macro")
weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
classification = classification_report(all_labels, all_preds)

# 保存评估结果到文件
report_path = "data/processed/MLP_Model_Report_Optimized.txt"

# 生成评估结果字符串
evaluation_report = (
    f"Accuracy: {accuracy:.4f}\n"
    f"Macro F1-Score: {macro_f1:.4f}\n"
    f"Weighted F1-Score: {weighted_f1:.4f}\n\n"
    f"Classification Report:\n{classification}\n"
)

# 将结果保存为文本文件
with open(report_path, "w") as file:
    file.write(evaluation_report)

print(f"Optimized model evaluation report saved to {report_path}")

# 定义模型保存路径
model_path = "models/MLP_Model.pth"

# 保存模型权重和相关信息
torch.save({
    'model_state_dict': model.state_dict(),  # 模型权重
    'input_size': input_size,
    'num_classes': num_classes,
    'class_mapping': class_mapping_filtered  # 保存类别映射
}, model_path)

print(f"Model saved to {model_path}")

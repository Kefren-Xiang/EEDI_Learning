import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 定义模型结构（与训练时一致）
class MLP(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 256)
        self.bn1 = torch.nn.BatchNorm1d(256)
        self.relu = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(0.3)

        self.fc2 = torch.nn.Linear(256, 128)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.dropout2 = torch.nn.Dropout(0.3)

        self.fc3 = torch.nn.Linear(128, 64)
        self.bn3 = torch.nn.BatchNorm1d(64)
        self.fc4 = torch.nn.Linear(64, num_classes)

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

# 加载模型
model_path = "models/MLP_Model.pth"
checkpoint = torch.load(model_path)
input_size = checkpoint['input_size']
num_classes = checkpoint['num_classes']
class_mapping = checkpoint['class_mapping']

model = MLP(input_size, num_classes)
model.load_state_dict(checkpoint['model_state_dict'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print("Model loaded successfully.")

# 加载测试数据
test_file_path = "data/raw/test.csv"
test_data = pd.read_csv(test_file_path)

# 构造 QuestionText+AnswerText 数据
records = []
for _, row in test_data.iterrows():
    for option, answer_col in zip(['A', 'B', 'C', 'D'], ['AnswerAText', 'AnswerBText', 'AnswerCText', 'AnswerDText']):
        # 跳过正确答案
        if row['CorrectAnswer'] == option:
            continue
        
        record = {
            "QuestionId_Answer": f"{row['QuestionId']}_{option}",
            "QuestionText+AnswerText": f"{row['QuestionText']} {row[answer_col]}"
        }
        records.append(record)

# 转换为 DataFrame
prediction_data = pd.DataFrame(records)

# 使用 TfidfVectorizer 将文本转化为数值特征
vectorizer = TfidfVectorizer(max_features=500)  # 提取最多 500 个关键特征
text_features = vectorizer.fit_transform(prediction_data['QuestionText+AnswerText']).toarray()

# 动态调整 PCA 的 n_components
scaler = StandardScaler()
n_components = min(input_size, text_features.shape[1], text_features.shape[0])  # 确保不超过样本或特征数
pca = PCA(n_components=n_components)

# 标准化和 PCA 降维
X = scaler.fit_transform(pca.fit_transform(text_features))

# 如果降维后的特征数少于模型输入的维度，补零扩展
if X.shape[1] < input_size:
    padding = input_size - X.shape[1]
    X = np.pad(X, ((0, 0), (0, padding)), mode='constant')

# 定义 PyTorch Dataset
class PredictionDataset(Dataset):
    def __init__(self, features):
        self.features = torch.tensor(features, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]

# 创建 DataLoader
prediction_dataset = PredictionDataset(X)
prediction_loader = DataLoader(prediction_dataset, batch_size=64, shuffle=False)

# 开始预测
predictions = []
with torch.no_grad():
    for features in prediction_loader:
        features = features.to(device)
        outputs = model(features)
        top_preds = torch.topk(outputs, 25, dim=1).indices  # 取Top 25 MisconceptionId
        predictions.extend(top_preds.cpu().numpy())

# 过滤和映射预测结果
def map_prediction(preds, class_mapping):
    mapped = []
    for p in preds:
        if p in class_mapping:
            mapped.append(str(class_mapping[p]))
        else:
            mapped.append("-1")  # 无法映射的类别设为 -1
    return " ".join(mapped)

prediction_data['MisconceptionId'] = [
    map_prediction(preds, class_mapping) for preds in predictions
]

# 保存为提交文件
submission_path = "data/processed/submission.csv"
submission_data = prediction_data[['QuestionId_Answer', 'MisconceptionId']]
submission_data.to_csv(submission_path, index=False)

print(f"Submission file saved to {submission_path}")

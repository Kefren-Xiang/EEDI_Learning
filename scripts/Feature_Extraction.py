from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

data_path = "data/processed/train_answers_cleaned.csv"

# 加载数据
df = pd.read_csv(data_path)

# 初始化 TF-IDF Vectorizer，限制最多 500 个关键词
tfidf_vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['QuestionText+AnswerText'])

# 将 TF-IDF 转化为 DataFrame
tfidf_feature_names = [f"TFIDF_{i+1}" for i in range(tfidf_matrix.shape[1])]
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_feature_names)

# 加载 BERT 模型和分词器
bert_model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
model = BertModel.from_pretrained(bert_model_name)

# 提取 BERT 的句向量特征
def get_bert_sentence_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    with torch.no_grad():
        outputs = model(**inputs)
    # 使用 [CLS] 标记的输出作为句向量
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return cls_embedding

# 对每一行计算 BERT 的句向量
bert_embeddings = []
for text in df['QuestionText+AnswerText']:
    bert_embeddings.append(get_bert_sentence_embedding(text))

# 转换为 DataFrame
bert_feature_names = [f"BERT_{i+1}" for i in range(768)]
bert_df = pd.DataFrame(bert_embeddings, columns=bert_feature_names)

# One-Hot 编码 ConstructId 和 SubjectId
encoder = OneHotEncoder(sparse_output=False)  # 使用正确的参数名称
construct_encoded = encoder.fit_transform(df[['ConstructId']])
subject_encoded = encoder.fit_transform(df[['SubjectId']])

construct_columns = [f"Construct_{i+1}" for i in range(construct_encoded.shape[1])]
subject_columns = [f"Subject_{i+1}" for i in range(subject_encoded.shape[1])]

construct_df = pd.DataFrame(construct_encoded, columns=construct_columns)
subject_df = pd.DataFrame(subject_encoded, columns=subject_columns)

# 合并所有特征
final_df = pd.concat([df[['AnswerId', 'MisconceptionId']], tfidf_df, bert_df, construct_df, subject_df], axis=1)

# 保存最终的特征表格
output_path = "data/processed/final_features.csv"
final_df.to_csv(output_path, index=False)

print(f"Final features saved to {output_path}")

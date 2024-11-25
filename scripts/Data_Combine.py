import pandas as pd

# 加载数据
train_path = "data/processed/train_ultimate.csv"
mapping_path = "data/processed/misconception_mapping_ultimate.csv"
output_path = "data/processed/train_answers_cleaned.csv"

train_df = pd.read_csv(train_path)
mapping_df = pd.read_csv(mapping_path)

# 重命名列方便合并
mapping_df.rename(columns={"MisconceptionId": "MisconceptionId", "MisconceptionName": "MisconceptionName"}, inplace=True)

# 定义函数：展开答案为基准的表格
def transform_to_answer_based_table(train_df, mapping_df):
    answer_rows = []

    for _, row in train_df.iterrows():
        for option, answer_text_col, misconception_col in zip(
            ["A", "B", "C", "D"],
            ["AnswerAText", "AnswerBText", "AnswerCText", "AnswerDText"],
            ["MisconceptionAId", "MisconceptionBId", "MisconceptionCId", "MisconceptionDId"]
        ):
            misconception_id = row[misconception_col]
            if pd.isna(misconception_id) or misconception_id < 0:
                continue  # 跳过正确答案和缺失值

            misconception_id = int(misconception_id)  # 转为整型
            misconception_name = mapping_df.loc[mapping_df["MisconceptionId"] == misconception_id, "MisconceptionName"].values
            misconception_name = misconception_name[0] if len(misconception_name) > 0 else "Unknown"

            answer_rows.append({
                "AnswerId": len(answer_rows) + 1,
                "QuestionText+AnswerText": f"{row['QuestionText']} {row[answer_text_col]}",
                "MisconceptionId": misconception_id,
                "MisconceptionName": misconception_name,
                "ConstructId": row["ConstructId"],
                "ConstructName": row["ConstructName"],
                "SubjectId": row["SubjectId"],
                "SubjectName": row["SubjectName"]
            })

    return pd.DataFrame(answer_rows)

# 转换数据
answer_based_df = transform_to_answer_based_table(train_df, mapping_df)

# 保存结果
answer_based_df.to_csv(output_path, index=False)
print(f"整合后的数据已保存至 {output_path}")

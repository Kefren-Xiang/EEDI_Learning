from transformers import BertTokenizer, BertModel

# 下载并保存到指定路径
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# 保存到本地文件夹
tokenizer.save_pretrained("bert_base_uncased")
model.save_pretrained("bert_base_uncased")

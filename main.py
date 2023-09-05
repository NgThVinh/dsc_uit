# from datasets import load_dataset

# dataset = load_dataset("squad_v2")


# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
# tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

# inputs = tokenizer("A step by step recipe to make bolognese pasta:", return_tensors="pt")
# outputs = model.generate(**inputs)
# print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

import pandas as pd

# Tạo DataFrame ví dụ
data = {'text': ['Text 1', 'Some text 2', 'Another text', 'Text with number 42'],
        'claim': ['Claim 1', 'Claim 2', 'Claim 3', 'Claim 4']}
df = pd.DataFrame(data)

# Kiểm tra các giá trị trong cột "text" có chứa số hay không
df['contains_number'] = df['text'].str.contains(r'\d')

# In ra DataFrame sau khi thêm cột "contains_number"
print(df)
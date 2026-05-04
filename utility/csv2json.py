import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('input.csv')

# 转换为 JSON（每行为一个对象的列表格式）
json_data = df.to_json(orient='records', lines=False, indent=2)

# 保存为 JSON 文件
with open('output.json', 'w', encoding='utf-8') as f:
    f.write(json_data)
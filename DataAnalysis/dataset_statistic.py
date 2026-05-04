import json
import os

# 定义需要统计的文件路径（相对或绝对）
base_dir = "/data5/storage4/wudauser05/yyc/RobustStressBench/domain_specific_model_specific/xsum"

file_paths = [
    os.path.join(base_dir, "train.json"),
    os.path.join(base_dir, "valid.json"),
    os.path.join(base_dir, "test.json"),
    os.path.join(base_dir, "SynonymReplacement/filter/test_perturb_5.json"),
    os.path.join(base_dir, "SynonymReplacement/filter/test_perturb_10.json"),
    os.path.join(base_dir, "SynonymReplacement/filter/test_perturb_15.json"),
    os.path.join(base_dir, "SynonymReplacement/filter/test_perturb_20.json"),
    os.path.join(base_dir, "polish/filter/test_low.json"),
    os.path.join(base_dir, "polish/filter/test_medium.json"),
    os.path.join(base_dir, "polish/filter/test_high.json"),
]

def count_json_lines(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return len(data)
            else:
                print(f"⚠️ Warning: {filepath} is not a JSON array. Skipping.")
                return 0
    except Exception as e:
        print(f"❌ Error reading {filepath}: {e}")
        return 0

print("📊 JSON 文件数据条数统计结果：\n")
total = 0
for path in file_paths:
    count = count_json_lines(path)
    filename = os.path.relpath(path, base_dir)  # 显示相对路径更简洁
    print(f"{filename:<30} : {count:>6} 条")
    total += count

print("-" * 50)
print(f"{'总计':<30} : {total:>6} 条")
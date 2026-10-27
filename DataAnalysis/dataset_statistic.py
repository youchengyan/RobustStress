import json
import os

# Path
base_dir = ""

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

print("📊 JSON statistic results：\n")
total = 0
for path in file_paths:
    count = count_json_lines(path)
    filename = os.path.relpath(path, base_dir)
    print(f"{filename:<30} : {count:>6} item")
    total += count

print("-" * 50)
print(f"{'sum':<30} : {total:>6} item")
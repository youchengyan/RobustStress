import spacy
import json
import random
from tqdm import tqdm
import argparse
import os

random.seed(42)
nlp = spacy.load('en_core_web_sm')

parser = argparse.ArgumentParser(description="Mark p% of ADJ/NOUN tokens (sampled from all candidates) with [[...]].")
parser.add_argument("--input_dir", type=str, required=True,
                    help="Directory containing 'test.json'")
args = parser.parse_args()

texts = []
labels = []
with open(os.path.join(args.input_dir, "test.json"), "r", encoding="utf-8") as f:
    data = json.load(f)
    for line in data:
        texts.append(line['text'])
        labels.append(line['label'])

L = [5, 10, 15, 20]

# 创建输出目录
output_dir = os.path.join(args.input_dir, "SynonymReplacement")
os.makedirs(output_dir, exist_ok=True)

for p in L:
    random.seed(42)
    result = []
    for i in tqdm(range(len(texts)), desc=f"p={p}%"):
        doc = nlp(texts[i])

        # 第一步：收集所有 ADJ/NOUN 的 token 及其索引
        candidate_indices = []
        tokens = []
        for idx, token in enumerate(doc):
            tokens.append(token.text)
            if token.pos_ == "ADJ" or token.pos_ == "NOUN":
                candidate_indices.append(idx)

        # 第二步：从候选中按比例采样要标记的数量
        if candidate_indices:
            num_to_mark = max(1, round(len(candidate_indices) * p / 100))  # 至少标记1个（可选）
            num_to_mark = min(num_to_mark, len(candidate_indices))  # 不超过总数
            marked_indices = set(random.sample(candidate_indices, num_to_mark))
        else:
            marked_indices = set()

        # 构建新文本：只对标记的索引加 [[...]]
        new_tokens = [
            f"[[{tokens[idx]}]]" if idx in marked_indices else tokens[idx]
            for idx in range(len(tokens))
        ]
        new_text = " ".join(new_tokens)
        result.append({"text": new_text, "label": labels[i]})

    # 保存结果
    with open(os.path.join(output_dir, f"test_perturb_{p}.json"), "w", encoding="utf-8") as w:
        json.dump(result, w, ensure_ascii=False, indent=2)
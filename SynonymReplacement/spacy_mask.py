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

# output dir
output_dir = os.path.join(args.input_dir, "SynonymReplacement")
os.makedirs(output_dir, exist_ok=True)

for p in L:
    random.seed(42)
    result = []
    for i in tqdm(range(len(texts)), desc=f"p={p}%"):
        doc = nlp(texts[i])

        # The first step
        candidate_indices = []
        tokens = []
        for idx, token in enumerate(doc):
            tokens.append(token.text)
            if token.pos_ == "ADJ" or token.pos_ == "ADV":
                candidate_indices.append(idx)

        # The second step
        if candidate_indices:
            num_to_mark = max(1, round(len(candidate_indices) * p / 100))
            num_to_mark = min(num_to_mark, len(candidate_indices))
            marked_indices = set(random.sample(candidate_indices, num_to_mark))
        else:
            marked_indices = set()

        new_tokens = [
            f"[[{tokens[idx]}]]" if idx in marked_indices else tokens[idx]
            for idx in range(len(tokens))
        ]
        new_text = " ".join(new_tokens)
        result.append({"text": new_text, "label": labels[i]})

    # save results
    with open(os.path.join(output_dir, f"test_perturb_{p}.json"), "w", encoding="utf-8") as w:
        json.dump(result, w, ensure_ascii=False, indent=2)
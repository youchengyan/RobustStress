import json
import random
import argparse
from pathlib import Path


def build_parallel_pairwise_dataset(
        input_json_path: str,
        output_json_path: str,
        seed: int = 42,
        max_pairs: int = None,
        shuffle: bool = True
):
    """
    construct parallel pairwise data：
    {
      "original": [human_text1, human_text2, ...],
      "sampled": [generated_text1, generated_text2, ...]
    }

    Args:
        input_json_path (str):  format [{"text": "...", "label": 0/1}, ...]
        output_json_path (str):
        seed (int):
        max_pairs (int or None):
        shuffle (bool):
    """
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    human_texts = []
    generated_texts = []

    for item in data:
        if 'text' not in item or 'label' not in item:
            raise ValueError(f"Each item must have 'text' and 'label'. Got: {item}")

        if item['label'] == 0:
            human_texts.append(item['text'])
        elif item['label'] == 1:
            generated_texts.append(item['text'])
        else:
            raise ValueError(f"Label must be 0 or 1. Got: {item['label']}")

    print(f"Found {len(human_texts)} human texts and {len(generated_texts)} generated texts.")

    if len(human_texts) == 0 or len(generated_texts) == 0:
        raise ValueError("Need at least one human and one generated text to build pairs!")

    n_possible = min(len(human_texts), len(generated_texts))
    n_pairs = min(max_pairs, n_possible) if max_pairs else n_possible
    print(f"Will create {n_pairs} pairs.")

    random.seed(seed)
    if shuffle:
        random.shuffle(human_texts)
        random.shuffle(generated_texts)

    original_list = human_texts[:n_pairs]
    sampled_list = generated_texts[:n_pairs]

    output_data = {
        "original": original_list,
        "sampled": sampled_list
    }

    output_path = Path(output_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"Saved parallel pairwise dataset to {output_json_path}")
    print(f"Keys: {list(output_data.keys())}")
    print(f"Length of each list: {len(original_list)}")

# python build_pairwise.py --input test.json --output data/test.json --seed 42
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build parallel pairwise dataset (two lists: original & sampled).")
    parser.add_argument("--input", required=True, help="Path to input JSON file (with 'text' and 'label')")
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--max_pairs", type=int, default=None, help="Max number of pairs to generate")
    parser.add_argument("--no_shuffle", action="store_true", help="Disable shuffling before pairing")

    args = parser.parse_args()

    build_parallel_pairwise_dataset(
        input_json_path=args.input,
        output_json_path=args.output,
        seed=args.seed,
        max_pairs=args.max_pairs,
        shuffle=not args.no_shuffle
    )
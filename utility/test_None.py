import json
import os
import argparse
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util


def main():
    parser = argparse.ArgumentParser(description="Filter synonym-replaced samples by SBERT similarity >= 0.95")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing 'test.json' and 'polish/' folder")
    parser.add_argument("--file_name", type=str, required=True,
                        help="low or medium or high")
    parser.add_argument("--threshold", type=str, required=True,
                        help="threshold")
    args = parser.parse_args()

    # Paths
    orig_file = os.path.join(args.input_dir, "test.json")
    perturb_dir = os.path.join(args.input_dir, "polish")


    # Process each perturbation file
    L = [args.file_name]
    for p in L:
        perturb_file = os.path.join(perturb_dir, f"test_{p}.json")
        if not os.path.exists(perturb_file):
            print(f"Warning: {perturb_file} not found. Skipping.")
            continue

        with open(perturb_file, "r", encoding="utf-8") as f:
            perturb_data = json.load(f)
        perturb_texts = [item["text"] for item in perturb_data]

        for idx, t in enumerate(perturb_texts):
            if t is None or not isinstance(t, str):
                print(f"Invalid input at {idx}: {repr(t)} (type: {type(t)})")


if __name__ == "__main__":
    main()

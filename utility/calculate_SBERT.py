import json
import os
import argparse
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util


def main():
    parser = argparse.ArgumentParser(description="Filter synonym-replaced samples by SBERT similarity >= 0.95")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing 'test.json' and 'SynonymReplacement/' folder")
    args = parser.parse_args()

    # Paths
    orig_file = os.path.join(args.input_dir, "test.json")
    perturb_dir = os.path.join(args.input_dir, "SynonymReplacement")

    # Load original test data
    with open(orig_file, "r", encoding="utf-8") as f:
        orig_data = json.load(f)
    orig_texts = [item["text"] for item in orig_data]
    labels = [item["label"] for item in orig_data]

    # Load SBERT model
    print("Loading SBERT model...")
    model = SentenceTransformer('/data5/storage4/wudauser05/yyc/all-MiniLM-L6-v2')

    # Encode original sentences once
    print("Encoding original sentences...")
    orig_embeddings = model.encode(orig_texts, convert_to_tensor=True, show_progress_bar=True)

    # Process each perturbation file
    L = [5, 10, 15, 20]
    for p in L:
        perturb_file = os.path.join(perturb_dir, f"test_perturb_{p}_synonym_replace.json")
        if not os.path.exists(perturb_file):
            print(f"Warning: {perturb_file} not found. Skipping.")
            continue

        with open(perturb_file, "r", encoding="utf-8") as f:
            perturb_data = json.load(f)
        perturb_texts = [item["text"] for item in perturb_data]

        # Safety check: same length?
        # if len(perturb_texts) != len(orig_texts):
        #     raise ValueError(f"Length mismatch for p={p}: orig={len(orig_texts)}, perturb={len(perturb_texts)}")

        # Encode perturbed sentences
        print(f"\nEncoding perturbed sentences (p={p}%)...")
        perturb_embeddings = model.encode(perturb_texts, convert_to_tensor=True, show_progress_bar=True)

        # Compute pairwise cosine similarities
        print(f"Computing similarities for p={p}%...")
        similarities = util.cos_sim(orig_embeddings, perturb_embeddings).diag()  # shape: [N]

        # Filter samples
        filtered_data = []
        for i in range(len(orig_texts)):
            sim = similarities[i].item()
            if sim >= 0.95:
                # Keep original label, perturbed text, optionally add similarity
                filtered_data.append({
                    "text": perturb_texts[i],
                    "label": labels[i]
                    # Optional: "similarity": round(sim, 4)
                })

        # Overwrite the perturbation file with filtered results
        # perturb_file = os.path.join(os.path.join(perturb_dir, "filter"), f"test_perturb_{p}.json")
        filtered_dir = os.path.join(perturb_dir, "filter")
        os.makedirs(filtered_dir, exist_ok=True)  # ✅ 自动创建 filter 文件夹（如果不存在）
        perturb_file = os.path.join(filtered_dir, f"test_perturb_{p}.json")
        with open(perturb_file, "w", encoding="utf-8") as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=2)

        print(f"p={p}%: kept {len(filtered_data)} / {len(orig_texts)} samples.")

    print("✅ All perturbation files filtered by SBERT similarity (≥0.95).")


if __name__ == "__main__":
    main()

import os
import argparse
import random
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from biscope_utils import data_generation
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2026, help="Random seed for reproducibility")
    parser.add_argument('--sample_clip', type=int, default=1000, help="Max token length for samples")
    parser.add_argument('--summary_model', type=str, default='none', help="Summary model key or 'none'")
    parser.add_argument('--detect_model', type=str, required=True, help="Detection model key")
    parser.add_argument('--train_dataset', type=str, required=True,
                        help='Format: {paraphrased or nonparaphrased}_{task}_{generative_model}')
    parser.add_argument('--test_dataset', type=str, required=True,
                        help='Format: {paraphrased or nonparaphrased}_{task}_{generative_model}')
    parser.add_argument('--output_dir', type=str, required=True, help="output_dir")
    parser.add_argument('--use_hf_dataset', type=bool, default=False, help="Load dataset from Hugging Face")
    parser.add_argument('--base_out_dir', type=str, default=False)
    args = parser.parse_args()

    if args.use_hf_dataset:
        print("Using Hugging Face datasets...")
    else:
        print("Using local datasets...")

    # Set seeds.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Create a base output directory that includes both train and test dataset strings.
    base_out_dir = os.path.join(args.base_out_dir)
    os.makedirs(base_out_dir, exist_ok=True)

    # Create separate subdirectories for train and test features.
    # If train and test datasets are identical, use the same directory.
    if args.train_dataset == args.test_dataset:
        train_dir = test_dir = base_out_dir
    else:
        train_dir = os.path.join(base_out_dir, "train")
        test_dir = os.path.join(base_out_dir, "test")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

    # Generate features for the training dataset.
    print("Generating train features...")
    data_generation(args, train_dir, args.train_dataset)

    # Load train features.
    with open(os.path.join(train_dir, f"human_features.pkl"), 'rb') as f:
        train_human = np.array(pickle.load(f))
    with open(os.path.join(train_dir, f"GPT_features.pkl"), 'rb') as f:
        train_gpt = np.array(pickle.load(f))

    data_generation(args, test_dir, args.test_dataset)
    with open(os.path.join(test_dir, f"human_features.pkl"), 'rb') as f:
        test_human = np.array(pickle.load(f))
    with open(os.path.join(test_dir, f"GPT_features.pkl"), 'rb') as f:
        test_gpt = np.array(pickle.load(f))
    train_feats = np.concatenate([train_human, train_gpt], axis=0)
    train_labels = np.concatenate([np.ones(len(train_gpt)), np.zeros(len(train_human))], axis=0)
    test_feats = np.concatenate([test_human, test_gpt], axis=0)
    test_labels = np.concatenate([np.ones(len(test_gpt)), np.zeros(len(test_human))], axis=0)
    clf = RandomForestClassifier(n_estimators=100, random_state=args.seed)
    clf.fit(train_feats, train_labels)
    preds = clf.predict(test_feats)
    pred_probs = clf.predict_proba(test_feats)[:, 1]  # 取正类（label=1）的概率

    precision = precision_score(test_labels, preds)
    recall = recall_score(test_labels, preds)
    f1 = f1_score(test_labels, preds)
    acc = accuracy_score(test_labels, preds)
    auroc = roc_auc_score(test_labels, pred_probs)  # 注意：这里用概率，不是 preds
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"Acc:       {acc:.4f}")
    print(f"AUROC:     {auroc:.4f}")

    # 保存结果
    result = {
        # 'test_labels': test_labels,
        # 'preds': preds,
        # 'pred_probs': pred_probs,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': acc,
        'auroc': auroc
    }

    basename = os.path.basename(args.test_dataset)  # → "test.json"
    file_name_without_ext = os.path.splitext(basename)[0]
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{file_name_without_ext}_BISCOPE_result.json")
    with open(output_path, 'w') as f:
        f.write(str(result))


# python biscope.py --detect_model llama2-7b --train_dataset /data5/storage4/wudauser05/yyc/RobustStressBench/domain_specific_model_specific/xsum/train.json --test_dataset /data5/storage4/wudauser05/yyc/RobustStressBench/domain_specific_model_specific/xsum/SynonymReplacement/filter/test_perturb_10.json --base_out_dir ./test_perturb_10
if __name__ == '__main__':
    main()

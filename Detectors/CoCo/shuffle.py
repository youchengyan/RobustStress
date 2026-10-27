import json
import random
import argparse
import os
import tempfile
import shutil


def shuffle_jsonl_inplace(file_path, seed=42):
    """
    shuffle JSONL files。

    Args:
        file_path (str): JSONL
        seed (int):
    """
    random.seed(seed)

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    non_empty_lines = [line for line in lines if line.strip()]

    random.shuffle(non_empty_lines)

    temp_fd, temp_path = tempfile.mkstemp(
        dir=os.path.dirname(file_path),
        prefix="tmp_shuffle_",
        suffix=".jsonl"
    )

    try:
        with os.fdopen(temp_fd, 'w', encoding='utf-8') as tmp_file:
            for line in non_empty_lines:
                if not line.endswith('\n'):
                    line += '\n'
                tmp_file.write(line)

        shutil.move(temp_path, file_path)
        print(f"Successfully shuffled and saved to: {file_path}")

    except Exception as e:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise e

# if you need to shuffle your own data
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shuffle a JSONL file in-place.")
    parser.add_argument("file", help="Path to the JSONL file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    shuffle_jsonl_inplace(args.file, seed=args.seed)
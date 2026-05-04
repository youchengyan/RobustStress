import json
import random
import argparse
import os
import tempfile
import shutil


def shuffle_jsonl_inplace(file_path, seed=42):
    """
    打乱 JSONL 文件的行顺序，并原地保存。

    Args:
        file_path (str): JSONL 文件路径
        seed (int): 随机种子，确保可复现
    """
    random.seed(seed)

    # 1. 读取所有行
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 2. 过滤空行（可选）
    non_empty_lines = [line for line in lines if line.strip()]

    # 3. 打乱顺序
    random.shuffle(non_empty_lines)

    # 4. 写入临时文件
    temp_fd, temp_path = tempfile.mkstemp(
        dir=os.path.dirname(file_path),
        prefix="tmp_shuffle_",
        suffix=".jsonl"
    )

    try:
        with os.fdopen(temp_fd, 'w', encoding='utf-8') as tmp_file:
            for line in non_empty_lines:
                # 确保每行以 \n 结尾
                if not line.endswith('\n'):
                    line += '\n'
                tmp_file.write(line)

        # 5. 原子替换原文件
        shutil.move(temp_path, file_path)
        print(f"Successfully shuffled and saved to: {file_path}")

    except Exception as e:
        # 出错时清理临时文件
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shuffle a JSONL file in-place.")
    parser.add_argument("file", help="Path to the JSONL file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    shuffle_jsonl_inplace(args.file, seed=args.seed)
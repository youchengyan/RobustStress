import json
import os
import tempfile
import shutil


def convert_labels_inplace(file_path):
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8', newline='')

    try:
        with open(file_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                line = line.rstrip('\n\r')
                if not line:
                    temp_file.write('\n')
                    continue

                data = json.loads(line)
                label = data.get("label")

                if label == 1:
                    data["label"] = "machine"
                elif label == 0:
                    data["label"] = "human"
                else:
                    pass  # or raise error

                temp_file.write(json.dumps(data, ensure_ascii=False) + '\n')

        temp_file.close()

        shutil.move(temp_file.name, file_path)
        print(f"Successfully updated labels in: {file_path}")

    except Exception as e:
        temp_file.close()
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        raise e


# if you need to convert labels
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to JSONL file")
    args = parser.parse_args()
    convert_labels_inplace(args.file)
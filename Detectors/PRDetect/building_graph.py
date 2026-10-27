import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "true"

import spacy
import torch
import json
import pickle
import numpy as np
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel
from scipy.sparse import csr_matrix
import argparse


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# loading English model
nlp = spacy.load("en_core_web_sm")
model = RobertaModel.from_pretrained(r"roberta-base").to(device)
tokenizer = RobertaTokenizer(r"roberta-base/vocab.json",
                             r"roberta-base/merges.txt", use_fast=False)
vocab_size = len(tokenizer)



def build_graph(json_texts):
    texts = list()
    y = list()
    for json_text in json_texts:
        texts.append(json_text['text'])
        label = json_text['label']
        y.append(label)
    y = torch.tensor(y, dtype=torch.float32)
    tokenized_sentences = list()
    all_token_embeddings = list()
    all_edge_index = list()
    for text in tqdm(texts):
        try:
            doc = nlp(text)
            tokenized_sentence = [token.text for token in doc]
            tokenized_sentences.append(tokenized_sentence)

            max_length = 512
            chunks = [tokenized_sentence[i:i + max_length] for i in range(0, len(tokenized_sentence), max_length)]
            chunk_outputs = []
            for chunk in chunks:
                token_ids = tokenizer.convert_tokens_to_ids(chunk)
                input_ids = torch.tensor(token_ids).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(input_ids)

                last_hidden_states = output.last_hidden_state
                token_embeddings = last_hidden_states[0]
                chunk_outputs.append(token_embeddings)
            token_embeddings = torch.cat(chunk_outputs, dim=0)
            # print(f'token_embeddings:{token_embeddings}')
            all_token_embeddings.append(token_embeddings)
            # print(len(tokenized_sentence))
            # print(token_embeddings.shape)
            node_relations = list()
            for word in doc:
                node_relations.append([word.i, word.head.i])

            edge0 = list()
            edge1 = list()
            for edge in node_relations:
                edge0.append(edge[0])
                edge1.append(edge[1])
            edge_index = torch.tensor([edge0, edge1], dtype=torch.long)
            all_edge_index.append(edge_index)

        except Exception as e:
            print(e)
    return all_token_embeddings, all_edge_index, y


def read_json(file_name, base_path):
    texts = list()
    with open(f"{base_path}/{file_name}.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        for item in data:
            texts.append(item)
    return texts


def save_pkl(file_name, base_path, all_token_embeddings, all_edge_index, y):
    graph_dir = os.path.join(base_path, "graph_data")
    os.makedirs(graph_dir, exist_ok=True)
    with open(f"{base_path}/graph_data/{file_name}.pkl", "wb") as f:
        pickle.dump({"all_token_embeddings": all_token_embeddings,
                     "all_edge_index": all_edge_index,
                     "y": y}, f)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Read JSON file from specified directory.")
    parser.add_argument(
        "--file_name",
        type=str,
        required=True,
        help="Name of the JSON file (without .json extension)"
    )
    parser.add_argument(
        "--base_path",
        type=str,
        default="RobustStressBench/domain_specific_model_specific/xsum",
        help="Base directory path containing JSON files"
    )

    args = parser.parse_args()

    all_token_embeddings, all_edge_index, y = build_graph(read_json(args.file_name, args.base_path))
    save_pkl(args.file_name, args.base_path, all_token_embeddings, all_edge_index, y)

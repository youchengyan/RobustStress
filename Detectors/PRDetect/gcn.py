# transformers==4.40.0
# 
# torch==1.11.0+cu113
# 
# torch-geometric==2.5.2
# 
# torch-scatter==2.0.9
# 
# torch-sparse==0.6.13

# 训练模型

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "true"
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
# from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
import time
import argparse
import pickle
import json

# 构建 GCN 模型
class GCN2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN2, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.fc = nn.Linear(output_dim, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = torch.mean(x, dim=0, keepdim=True)
        return torch.sigmoid(x)


def main():
    parser = argparse.ArgumentParser(description="Load graph data from .pkl files.")

    # 添加命令行参数
    parser.add_argument(
        "--train_path",
        type=str,
        default="./graph_data/train.pkl",
        help="Path to the training .pkl file (default: ./graph_data/train.pkl)"
    )
    parser.add_argument(
        "--val_path",
        type=str,
        default="./graph_data/val.pkl",
        help="Path to the validation .pkl file (default: ./graph_data/val.pkl)"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        required=True,
        help="Name of the test file (e.g., 'test' -> loads ./graph_data/test.pkl)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./xsum",
        help="Base directory for graph data (used to construct test path if not using full path)"
    )

    args = parser.parse_args()
    with open(args.train_path, "rb") as f:
        hc3_train = pickle.load(f)
    with open(args.val_path, "rb") as f:
        hc3_val = pickle.load(f)

    basename = os.path.basename(args.test_file)  # → "test.pkl"
    file_name_without_ext = os.path.splitext(basename)[0]

    seed = 2026
    dataset_name = 'hc3'
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = 768  # 输入维度
    hidden_dim = 512  # 隐藏层维度
    hidden_dim2 = 256  # 隐藏层维度
    hidden_dim3 = 128  # 隐藏层维度
    output_dim = 64  # 输出类别数
    gcnmodel = GCN2(input_dim, hidden_dim2, output_dim).to(device)
    optimizer = optim.Adam(gcnmodel.parameters(), lr=0.0001)
    criterion = nn.BCELoss()

    train_len = len(hc3_train['y'])
    print(f'train_len:{train_len}')
    val_len = len(hc3_val['y'])
    epochs = 40
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    val_max_acc = -1
    # start_time = time.time()
    for epoch in range(epochs):
        # 训练集
        gcnmodel.train()
        epoch_loss = 0.0
        correct_predictions = 0
        for i in tqdm(range(train_len), f"epoch: {epoch + 1}, Training"):
            data = Data(x=hc3_train['all_token_embeddings'][i], edge_index=hc3_train['all_edge_index'][i],
                        y=hc3_train['y'][i]).to(device)
            optimizer.zero_grad()
            outputs = gcnmodel(data)
            loss = criterion(outputs, data.y.float().view(-1, 1))
            # print(loss)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            predictions = (outputs >= 0.5).long()
            correct_predictions += (predictions == data.y.view(-1, 1)).sum().item()
        epoch_loss /= train_len
        epoch_acc = correct_predictions / train_len
        print(f"epoch: {epoch + 1}, train_loss: {epoch_loss}, train_acc: {epoch_acc}")
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)
        #     # 验证集
        gcnmodel.eval()
        epoch_loss = 0.0
        correct_predictions = 0
        all_predictions = []
        with torch.no_grad():
            for i in tqdm(range(val_len), f"epoch: {epoch + 1}, Validation"):
                data = Data(x=hc3_val['all_token_embeddings'][i], edge_index=hc3_val['all_edge_index'][i],
                            y=hc3_val['y'][i]).to(device)
                outputs = gcnmodel(data)
                loss = criterion(outputs, data.y.float().view(-1, 1))
                epoch_loss += loss.item()
                predictions = (outputs >= 0.5).long()
                all_predictions.append(predictions)
                correct_predictions += (predictions == data.y.view(-1, 1)).sum().item()
        epoch_loss /= val_len
        epoch_acc = correct_predictions / val_len
        print(f"epoch: {epoch + 1}, val_loss: {epoch_loss}, val_acc: {epoch_acc}")
        val_loss.append(epoch_loss)
        val_acc.append(epoch_acc)

        tag = 3
        if epoch_acc >= val_max_acc:
            val_max_acc = epoch_acc
            tag = 3
            torch.save(gcnmodel.state_dict(), f'./model/{file_name_without_ext}_gcn_model_{seed}.pth')
        else:
            tag -= 1
            if tag == 0:
                break
    # 测试

    with open(args.test_file, "rb") as f:
        hc3_test = pickle.load(f)
    test_len = len(hc3_test['y'])

    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

    test_gcnmodel = GCN2(input_dim, hidden_dim2, output_dim).to(device)
    test_gcnmodel.load_state_dict(torch.load(f'./model/{file_name_without_ext}_gcn_model_{seed}.pth'))
    test_gcnmodel.eval()
    test_loss = 0.0
    correct_predictions = 0
    test_pres = list()
    start_time = time.time()
    with torch.no_grad():
        for i in tqdm(range(test_len), f"Test"):
            data = Data(x=hc3_test['all_token_embeddings'][i], edge_index=hc3_test['all_edge_index'][i],
                        y=hc3_test['y'][i]).to(device)
            outputs = test_gcnmodel(data)
            test_pres.append(outputs.item())
            loss = criterion(outputs, data.y.float().view(-1, 1))
            test_loss += loss.item()
            predictions = (outputs >= 0.5).long()
            correct_predictions += (predictions == data.y.view(-1, 1)).sum().item()

    y_pred = [1 if prob >= 0.5 else 0 for prob in test_pres]
    y_true = hc3_test['y'].view(-1, 1)

    test_loss /= test_len
    test_acc = correct_predictions / test_len
    test_f1 = f1_score(y_true, y_pred)
    print(f"test_loss: {test_loss}, test_acc: {test_acc}, test_f1: {test_f1}")
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    print(f'precision:{precision}, recall:{recall}')

    auc = roc_auc_score(hc3_test['y'], test_pres)
    print(f"roc_auc:{auc}")

    result = {
        "roc_auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": test_f1,
        "acc": test_acc,
        "seed": seed
    }
    output_dir = args.output_dir
    file_name = f"{file_name_without_ext}_PRDetect_result.json"
    output_path = os.path.join(output_dir, file_name)

    # 自动创建目录（exist_ok=True 表示如果已存在也不报错）
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    main()

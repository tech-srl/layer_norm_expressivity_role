import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
from enum import Enum, auto
from numpy.random import default_rng
import math
import os
import json
import argparse
from sklearn.model_selection import train_test_split
from torch_scatter import scatter
import wandb
from transformers import get_linear_schedule_with_warmup
rng = None


def partition(n, max_i, slices):
    if n == 0:
        yield []
        return
    if slices - 1 == 0:
        yield [n]
        return

    arr = np.arange(max_i - 1, n) + 1
    rng.shuffle(arr)
    for i in arr:
        for p in partition(n - i, i, slices - 1):
            yield [i] + p


def get_partition_generator(seq_len, num_classes):
    while True:
        gen = partition(seq_len, 1, min(seq_len, num_classes))
        for part in gen:
            yield part


def genrate_for_partition(part, seq_len, num_classes, num_samples):
    res = np.zeros((num_samples, seq_len))
    for i in range(num_samples):
        perm = rng.permutation(num_classes)
        last_idx = 0
        for j, elem in enumerate(part):
            res[i, last_idx: last_idx + elem] = perm[j]
            last_idx += elem
    return res


def gen_X(num_samples, seq_len, num_classes, per_partition=10, min_delta=10):
    # num of partition ~ seq_len^2
    X = []
    for part in partition(seq_len, 1, min(seq_len, num_classes)):
        if len(part) == 1 or part[-1] == part[-2]:  # two classes are the majorities
            continue
        if part[-1] - part[-2] < min_delta: # ensure the majority class is significat
            continue
        X.append(genrate_for_partition(part, seq_len, num_classes, per_partition))
        if len(X) * per_partition > num_samples:
            break
    # (num_partitions, per_partition, seq_len)
    X = torch.LongTensor(np.array(X))
    # (~num_partitions, seq_len)
    X = X.reshape(-1, seq_len)
    return X


def gen_data(num_samples, seq_len, num_classes, min_delta):
    # (num_samples, seq_len)
    X = gen_X(num_samples, seq_len, num_classes, min_delta=min_delta)
    # (num_samples, num_classes)
    counts = scatter(src=torch.ones_like(X), index=X)
    # (num_samples)
    y = torch.argmax(counts, dim=-1)
    # (num_samples, 1)
    y = y.unsqueeze(dim=-1)
    # (num_samples, seq_len)
    y = y.repeat_interleave(seq_len, dim=-1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, stratify=y[:, -1])
    classes = torch.arange(num_classes).unsqueeze(dim=0)
    return X_train, X_test, y_train, y_test, classes

class Id(nn.Module):
    def __init__(self, dim):
        super(Id, self).__init__()
    def forward(self, x):
        return x

class LayerNormWithoutProj(nn.Module):
    def __init__(self, dim):
        super(LayerNormWithoutProj, self).__init__()
        self.factor = dim**0.5
    def forward(self, x):
        # return x
        return nn.functional.normalize(x, dim=-1) / self.factor

class Model(nn.Module):
    def __init__(self, num_classes, hidden_dim, num_head, layer_norm):
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.emb_layer = nn.Embedding(num_classes, hidden_dim)
        self.transformer = torch.nn.TransformerEncoderLayer(d_model=hidden_dim,
                                                            nhead=num_head,
                                                            dim_feedforward=hidden_dim,
                                                            dropout=0.0,
                                                            layer_norm_eps=0.0,
                                                            batch_first=True,
                                                            norm_first=True)
        if layer_norm is LAYER_NORM.WITH_PROJ:
            self.transformer.norm1.register_parameter('weight', None)
            self.transformer.norm1.register_parameter('bias', None)
        elif layer_norm is LAYER_NORM.WITHOUT_PROJ:
            self.transformer.norm1 = LayerNormWithoutProj(hidden_dim)
        else:
            self.transformer.norm1 = Id(hidden_dim)

        self.transformer.norm1.register_parameter('weight', None)
        self.transformer.norm1.register_parameter('bias', None)
        self.transformer.self_attn.register_parameter('in_proj_bias', None)

    def forward(self, input):
        batch_size, seq_len = input.shape
        # (batch_size, seq_len, hidden_dim)
        emb = self.emb_layer(input)
        # (batch_size, seq_len, hidden_dim)
        transformer_out = self.transformer(emb)
        # (batch_size, seq_len, hidden_dim)
        queries = self.transformer.norm1(emb)# [:,1:,:]
        # (hidden_dim [out features], hidden_dim [in features])
        Q, K, V = self.transformer.self_attn.in_proj_weight.chunk(3, dim=0)
        # (batch_size, seq_len, hidden_dim)
        qQK = queries @ Q.T @ K / math.sqrt(self.hidden_dim)
        # (batch_size, seq_len, num_classes)
        logits = transformer_out @ self.emb_layer.weight.T
        keys = queries.detach()
        return logits, qQK, keys

class data:
    def __init__(self):
        self.X = []
        self.y = []
        self.prediction = []
        self.angle = []
        self.angle_mean = 0.0
        self.angle_std = 0.0
        self.total_loss = 0.0
        self.loss = 0.0
        self.acc = 0.0
        self.correct = 0
        self.total = 0

        self.failed_X = []
        self.failed_y_hat = []
        self.success_X = []
        self.success_y_hat = []

    def process(self):
        self.acc = self.correct / self.total
        self.loss = self.total_loss / self.total

        self.angle = torch.cat(self.angle, dim=0)
        self.angle_mean = self.angle.mean().item()
        self.angle_std = self.angle.std().item()

        self.failed_X = torch.cat(self.failed_X, dim=0)
        self.failed_y_hat = torch.cat(self.failed_y_hat, dim=0)
        self.success_X = torch.cat(self.success_X, dim=0)
        self.success_y_hat = torch.cat(self.success_y_hat, dim=0)

def run_experiment(num_samples, seq_len, num_classes, hidden_dim, num_head, epochs, lr, batch_size, min_delta, layer_norm):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    X_train, X_test, y_train, y_test, classes = gen_data(num_samples, seq_len, num_classes, min_delta)
    wandb.config['training_exampels'] = len(X_train)
    wandb.config['test_exampels'] = len(X_test)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                pin_memory=True, num_workers=1)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                                pin_memory=True, num_workers=1)

    model = Model(num_classes, hidden_dim, num_head, layer_norm).to(device)
    wandb.config['num_parameters'] =  sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    num_steps = math.ceil(len(train_dataset) / batch_size) * 10000
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, num_steps)
    criterion = nn.CrossEntropyLoss()
    cos = torch.nn.CosineSimilarity(dim=-1)
    ones = torch.ones(hidden_dim).to(device)
    classes = classes.to(device)

    for i in range(epochs):
        train_data = data()
        epoch_queries = []
        for j, (X, y) in enumerate(train_loader):
            X = X.to(device)
            # (batch_size, seq_len)
            y = y.to(device)
            # (batch_size * seq_len)
            y_flattened = y.reshape(-1)
            current_batch_size, seq_len = X.shape
            # logits: (batch_size, seq_len, num_classes)
            # qQK, keys, values, emb, out : (batch_size, seq_len, hidden_dim)
            logits, qQK, keys = model(X)

            # (batch_size * seq_len, num_classes)
            logits_flattened = logits.reshape(current_batch_size * seq_len, -1)

            train_cosine = cos(qQK, ones)
            train_data.angle.append(90 - torch.abs(torch.rad2deg(torch.acos(train_cosine)) - 90))


            loss = criterion(logits_flattened, y_flattened)
            train_data.total_loss += loss.item() * len(y_flattened)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            # (batch_size * seq_len)
            train_pred_flattened = logits_flattened.argmax(dim=-1)
            # train_data.prediction.append(logits.argmax(dim=-1))
            train_data.correct += (train_pred_flattened == y_flattened).sum().item()
            train_data.total += len(y_flattened)
            
            # (batch_size, seq_len)  
            train_pred = logits.argmax(dim=-1)
            # (batch_size)
            mask = torch.all(train_pred == y, dim=-1)
            train_data.failed_X.append(X[~mask])
            train_data.failed_y_hat.append(train_pred[~mask])
            train_data.success_X.append(X[mask])   
            train_data.success_y_hat.append(train_pred[mask])

        train_data.process()

        with torch.no_grad():
            # model.eval() causes a bug and thus we are not using model.eval() https://github.com/pytorch/pytorch/issues/88669
            # it doesn't really matters since we are not using dropout or something like that
            test_data = data()
            for j, (X, y) in enumerate(test_loader):
                X = X.to(device)
                y = y.to(device)
                current_batch_size, seq_len = X.shape
                logits, qQK, keys = model(X)

                cosine = cos(qQK, ones)

                test_data.angle.append(90 - torch.abs(torch.rad2deg(torch.acos(cosine)) - 90))

                test_pred_flattened = logits.reshape(current_batch_size * seq_len, -1).argmax(dim=-1)

                y_flattened = y.reshape(-1)
                test_data.correct += (test_pred_flattened == y_flattened).sum().item()
                test_data.total += len(y_flattened)

                # (batch_size, seq_len)  
                test_pred = logits.argmax(dim=-1)
                # (batch_size)
                mask = torch.all(test_pred == y, dim=-1)
                test_data.failed_X.append(X[~mask])
                test_data.failed_y_hat.append(test_pred[~mask])
                test_data.success_X.append(X[mask])   
                test_data.success_y_hat.append(test_pred[mask])

            test_data.process()

        print(
            f"Epoch: {i:>3} loss: {train_data.loss:>6.4f}  train acc: {train_data.acc:>6.4f}  test acc: {test_data.acc:>6.4f} "
            f"train angle: {train_data.angle_mean:>6.4f}±{train_data.angle_std:>6.4f}")


        wandb.log({"loss": train_data.loss,
                    "train_acc": train_data.acc,
                    "test_acc": test_data.acc,
                    "train_angle_hist": train_data.angle.reshape(-1),
                    "test_angle_hist": test_data.angle.reshape(-1),
                    "train_angle_mean": train_data.angle_mean,
                    "train_angle_std": train_data.angle_std,
                    "test_angle_mean": test_data.angle_mean,
                    "test_angle_std": test_data.angle_std,
                    "learning_rate": [group['lr'] for group in optimizer.param_groups][0],
                    })
        if test_data.acc >= 1.0:
            print("test acc >= 1.0")
            break

    # (1, num_classes,     hidden_dim)
    _, queries, keys = model(classes)
    if hidden_dim == 3:
        wandb.run.summary['queries'] = queries
        wandb.run.summary['keys'] = keys
        with open(f'{layer_norm}_keys_queries.json', 'w') as f:
            json.dump({'keys': keys, 'queries': queries}, f)


def set_seed(seed: int = 11):
    global rng
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    rng = default_rng()

def main():
    args = parse_args()
    print(args)
    group = args.pop('group')
    mode = args.pop('mode')
    notes = args.pop('notes')
    wandb.init(config=args, group=group, project="majority", mode=mode, notes=notes)
    wandb.config['SLURM_JOB_ID'] = os.getenv('SLURM_JOB_ID')
    wandb.config['SLURM_ARRAY_JOB_ID'] = os.getenv('SLURM_ARRAY_JOB_ID')

    set_seed(wandb.config.seed)

    run_experiment(num_samples=wandb.config.num_samples,
                   seq_len=wandb.config.seq_len,
                   num_classes=wandb.config.num_classes,
                   hidden_dim=wandb.config.hidden_dim,
                   num_head=wandb.config.num_head,
                   epochs=wandb.config.epochs,
                   lr=wandb.config.lr,
                   batch_size=wandb.config.batch_size,
                   min_delta=wandb.config.data_min_delta,
                   layer_norm=LAYER_NORM.from_string(wandb.config.layer_norm))

class LAYER_NORM(Enum):
    WITH_PROJ = auto()
    WITHOUT_PROJ = auto()
    NONE = auto()

    @staticmethod
    def from_string(s):
        try:
            return LAYER_NORM[s]
        except KeyError:
            raise ValueError()
    
    def __str__(self):
        if self is LAYER_NORM.WITH_PROJ:
            return "WITH_PROJ"
        elif self is LAYER_NORM.WITHOUT_PROJ:
            return "WITHOUT_PROJ"
        else:
            return "NONE"

def none_or_str(value):
    if value == 'None':
        return None
    return value

def parse_args():
    parser = argparse.ArgumentParser(description='Majority Experiment')
    parser.add_argument('--num_samples', default=100000, type=int)
    parser.add_argument('--seq_len', default=50, type=int)
    parser.add_argument('--num_classes', default=20, type=int) # 20
    parser.add_argument('--hidden_dim', default=8, type=int)
    parser.add_argument('--num_head', default=1, type=int)
    parser.add_argument('--epochs', default=5, type=int)     
    parser.add_argument('--lr', default=0.001, type=float)    
    parser.add_argument('--batch_size', default=6000, type=int)
    parser.add_argument('--data_min_delta', default=6, type=int)   # 5 
    parser.add_argument('--seed', default=11, type=int)
    parser.add_argument('--group', default=None, type=str)
    parser.add_argument('--mode', default='disabled', type=none_or_str)
    parser.add_argument('--notes', default=None, type=none_or_str)
    parser.add_argument("--layer_norm", default=LAYER_NORM.WITH_PROJ,
                        type=LAYER_NORM.from_string, choices=list(LAYER_NORM)) 
    args = parser.parse_args()
    return vars(args)
                        

if __name__ == '__main__':
    main()

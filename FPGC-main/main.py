from __future__ import print_function, division
from sklearn.cluster import KMeans
from torch.optim import Adam
from utils import *
from model import *
import argparse
import numpy as np
from dataset_loader import DataLoader
import random
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train.')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='The value of weight decay.')
parser.add_argument('--k', type=int, default=3, help='Aggregation layers.')
parser.add_argument('--lamb', type=float, default=1, help='Balance parameter.')
parser.add_argument('--beta1', type=int, default=60, help='Number of fields.')
parser.add_argument('--beta2', type=int, default=100, help='Number of essential features.')
parser.add_argument('--hiddenDim', type=int, default=500, help='Dim of hidden layer.')
parser.add_argument('--dataset', type=str, default='cora', help='Type of dataset.')
parser.add_argument('--device', type=int, default=0, help='Train on which gpu.')
args = parser.parse_args()

def compute_fea(X, n_slices, device):
    rows, cols = X.shape
    slice_cols = cols // n_slices
    remainder = cols % n_slices
    zeros_needed = 0 if remainder == 0 else n_slices - remainder
    if zeros_needed > 0:
        zeros = torch.zeros((rows, zeros_needed), device = device)
        X = torch.cat((X, zeros), dim=1)
        slices = [row.split(slice_cols+1, dim=0) for row in X]
        slices = [torch.stack(slice_list, dim=0) for slice_list in slices]
    else:
        slices = [row.split(slice_cols, dim=0) for row in X]
        slices = [torch.stack(slice_list, dim=0) for slice_list in slices]
    return slices

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def normalize_matrix(A, eps=1e-12):
    D = np.sum(A, axis=1) + eps
    D = np.power(D, -0.5)
    D[np.isinf(D)] = 0
    D[np.isnan(D)] = 0
    D = np.diagflat(D)
    A = D.dot(A).dot(D)
    return A

def sim(z1, z2, tau = 1):
    z1_norm = torch.norm(z1, dim=-1, keepdim=True)
    z2_norm = torch.norm(z2, dim=-1, keepdim=True)
    dot_numerator = torch.mm(z1, z2.t())
    dot_denominator = torch.mm(z1_norm, z2_norm.t()) + EPS
    sim_matrix = torch.exp(dot_numerator / dot_denominator / tau)
    return sim_matrix

if __name__ == '__main__':
    EPS = 1e-15
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    dataset = args.dataset
    data = DataLoader(dataset.lower())
    X = np.array(data.x)
    y = np.array(data.y)
    A = sp.coo_matrix((np.ones(data.num_edges), (data.edge_index[0].numpy(), data.edge_index[1].numpy())),
                      shape=(data.num_nodes, data.num_nodes))
    A = A.toarray()
    n_node, n_input = A.shape[0], X.shape[1]
    n_clusters = (y.max() + 1).item()
    data = data.to(device)
    X_ = X.copy()

    # normalized matrix
    I = np.eye(A.shape[0])
    A= A + I
    A = normalize_matrix(A)
    A = torch.from_numpy(A)
    X = torch.from_numpy(X)
    A = A.to(device).float()
    X = X.to(device).float()
    X_ = X.clone()
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    for i in range(args.k):
        X = torch.mm(A, X)
    n_num = args.beta1
    n_num_sum = int(n_num * (n_num - 1) / 2)
    X3 = compute_fea(X, n_num, device)
    X2 = torch.zeros(n_node, n_num_sum).to(device)
    for i in range(n_node):
        tmp = torch.mm(X3[i], X3[i].t())
        for i1 in range(n_num - 1):
            for j1 in range(i1 + 1, n_num):
                a = int((n_num - 1) * i1 - (i1 * (i1 + 1) / 2) + j1 - 1)
                X2[i][a] = tmp[i1][j1]
    X2 = X2.to(device).float()
    acc_list = []
    nmi_list = []
    for seed in range(10):
        set_seed(seed)
        gcn_sc = None
        best_acc = 0.
        best_nmi = 0.
        best_emb = None
        best_epoch = 0
        model = FPGC(args.hiddenDim, n_input=n_input, dim = args.beta2, num = n_num_sum).to(device)
        optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        for epoch in tqdm(range(args.epochs)):
            embed, embed2 = model(X,X_,X2,data,device)
            embed_f = (embed + embed2)/2
            y_pred = kmeans.fit_predict(embed_f.data.cpu().numpy())
            acc_cur, nmi_cur = eva(y, y_pred, str(epoch))
            if best_acc < acc_cur:
                best_acc = acc_cur
                best_emb = embed
                best_nmi = nmi_cur
            gcn_sc = embed @ embed2.T
            re_loss = F.mse_loss(gcn_sc, A)
            pos = torch.eye(len(embed)).to_sparse().to(device)
            matrix_1 = sim(embed, embed2)
            matrix_2 = matrix_1.t()
            matrix_1 = matrix_1 / (torch.sum(matrix_1, dim=1).view(-1, 1) + EPS)
            lori_1 = -torch.log(matrix_1.mul(pos.to_dense()).sum(dim=-1)).mean()
            matrix_2 = matrix_2 / (torch.sum(matrix_2, dim=1).view(-1, 1) + EPS)
            lori_2 = -torch.log(matrix_2.mul(pos.to_dense()).sum(dim=-1)).mean()
            cl_loss = (lori_1 + lori_2) / 2
            loss = args.lamb * cl_loss + re_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("Seed:", seed, "ACC:", best_acc, "NMI:", best_nmi)
        acc_list.append(best_acc)
        nmi_list.append(best_nmi)
    acc_list = np.array(acc_list)
    nmi_list = np.array(nmi_list)
    print("Total ACC:", round(acc_list.mean(), 4), "STD:", round(acc_list.std(), 4), "Total NMI:", round(nmi_list.mean(), 4), "STD:", round(nmi_list.std(), 4))
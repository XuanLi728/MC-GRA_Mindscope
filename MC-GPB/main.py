import argparse
import os
import warnings
from copy import deepcopy

import numpy as np
import torch.nn.functional as F
from mind_dataset import Dataset

import mindspore

import random

from mind_dataset import Dataset
from mindspore import context
from mindspore import ops
from mindspore import Tensor


# from models.gat import GAT, embedding_gat
from models.mind_gcn import GCN, embedding_GCN  # gcn_hetero
# from models.graphsage import embedding_graphsage, graphsage
from sklearn.metrics import auc, average_precision_score, roc_curve
from topology_attack import PGDAttack
from utils import *

warnings.filterwarnings('ignore')


def test(adj, features, labels, idx_test, victim_model):
    adj, features, labels = to_tensor(adj, features, labels, device=device)
    victim_model.eval()
    adj_norm = normalize_adj_tensor(adj)
    output = victim_model(features, adj_norm)[0]
    acc_test = accuracy(output[idx_test], labels[idx_test])
    return acc_test


    Z = ops.L2Normalize(Z, axis=1)
    Z = ops.matmul(Z, Z.t())
    adj = ops.relu(Z-ops.eye(Z.shape[0]))
    return adj


def dot_product_decode(Z):
    if args.dataset in ['polblogs', 'brazil', 'usair']:
        Z = ops.L2Normalize(axis=1)(Z)
    Z = ops.matmul(Z, Z.t())
    adj = ops.relu(Z-ops.eye(Z.shape[0]))
    return adj


def preprocess_Adj(adj, feature_adj):
    n = len(adj)
    cnt = 0
    adj = adj.numpy()
    feature_adj = feature_adj.numpy()
    for i in range(n):
        for j in range(n):
            if feature_adj[i][j] > 0.14 and adj[i][j] == 0.0:
                adj[i][j] = 1.0
                cnt += 1
    return torch.FloatTensor(adj)


# def transfer_state_dict(pretrained_dict, model_dict):
#     state_dict = {}
#     for k, v in pretrained_dict.items():
#         if k in model_dict.keys():
#             state_dict[k] = v
#     return state_dict
def transfer_state_dict(pretrained_dict, model_dict):
    state_dict = {}
    for k, v in pretrained_dict:
        if k in model_dict.keys():
            state_dict[k] = v
    return state_dict


def metric(ori_adj, inference_adj, idx):

    real_edge = ori_adj[idx, :][:, idx].reshape(-1)
    pred_edge = inference_adj[idx, :][:, idx].reshape(-1)

    fpr, tpr, threshold = roc_curve(real_edge, pred_edge)
    AUC_adj = auc(fpr, tpr)
    index = np.where(real_edge == 0)[0]
    index_delete = np.random.choice(index, size=int(
        len(real_edge)-2*np.sum(real_edge)), replace=False)
    real_edge = np.delete(real_edge, index_delete)
    pred_edge = np.delete(pred_edge, index_delete)
    return AUC_adj


def homo_hetero_edge_extractor(adj, y):
    homo_adj = np.zeros_like(adj)
    hetero_adj = np.zeros_like(adj)

    for row in range(adj.shape[0]):
        tmp = np.nonzero(adj[row])
        if len(tmp) == 0:
            continue
        cols = tmp[0]
        for col in cols:
            if y[row] == y[col]:
                homo_adj[row][col] = 1
            else:
                hetero_adj[row][col] = 1

    return homo_adj, hetero_adj


def adj_auc(ori_adj, inference_adj, y):

    def auc_ap_calc(gt_adj, pred_adj):
        real_edge = gt_adj.reshape(-1)
        pred_edge = pred_adj.reshape(-1)
        fpr, tpr, _ = roc_curve(real_edge, pred_edge)
        return auc(fpr, tpr), average_precision_score(real_edge, pred_edge)

    homo, hetero = homo_hetero_edge_extractor(ori_adj, y)
    homo_auc, homo_ap = auc_ap_calc(homo, inference_adj)
    hetero_auc, hetero_ap = auc_ap_calc(hetero, inference_adj)

    return homo_auc, homo_ap, hetero_auc, hetero_ap


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to optimize in GraphMI attack.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='cora',
                    choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed', 'AIDS', 'usair', 'brazil', 'enzyme', 'ogb_arxiv'], help='dataset')
parser.add_argument('--density', type=float, default=1.0,
                    help='Edge density estimation')
parser.add_argument('--model', type=str, default='PGD',
                    choices=['PGD', 'min-max'], help='model variant')
parser.add_argument('--nlabel', type=float, default=0.1)

parser.add_argument('--arch', type=str, default='gcn')
parser.add_argument('--nlayer', type=int, default=2)
parser.add_argument('--MI_type', type=str, default='KDE')

parser.add_argument('--layer_MI', nargs='+',
                    help='the layer MI constrain')

parser.add_argument('--layer_inter_MI', nargs='+',
                    help='the inter-layer MI constrain')

parser.add_argument('--aug_pe', type=float,
                    default='proability of augmentation')

parser.add_argument('--device', type=str, default='cuda:0')

args = parser.parse_args()

setup_seed(args.seed)

# global device 
# device= torch.device(args.device if torch.cuda.is_available() else "cpu")
# if device != 'cpu':
#     torch.cuda.manual_seed(args.seed)

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

#print(device)
data = Dataset(root='dataset', name=args.dataset, setting='GCN')
adj, features, labels, init_adj = data.adj, data.features, data.labels, data.init_adj

idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

idx_attack = np.arange(adj.shape[0])
num_edges = int(0.5 * args.density * adj.sum() /
                adj.shape[0]**2 * len(idx_attack)**2)
adj, features, labels = preprocess(
    adj, features, labels, preprocess_adj=False, onehot_feature=False)
feature_adj = dot_product_decode(features)
init_adj = ops.Tensor(init_adj.todense())

if args.arch == 'gcn':
    base_model = GCN(
        nfeat=features.shape[1],
        nclass=labels.max().item() + 1,
        nhid=16,
        idx_train=idx_train,
        nlayer=args.nlayer,
        dropout=0.5,
        weight_decay=5e-4,
    )
# elif args.arch == 'sage':
#     base_model = graphsage(
#         nfeat=features.shape[1],
#         nclass=labels.max().item() + 1,
#         nhid=16,
#         nlayer=args.nlayer,
#         dropout=0.5,
#         weight_decay=5e-4,
#         device=device
#     )
# elif args.arch == 'gat':
#     base_model = GAT(
#         nfeat=features.shape[1],
#         nhid=16,
#         nclass=labels.max().item() + 1,
#         nheads=4,
#         dropout=0.5,
#         alpha=0.1,
#         nlayer=args.nlayer,
#         device=device
#     )
else:
    print('Unknown model arch')


def objective(param):

    victim_model = deepcopy(base_model)
    # victim_model = victim_model .to(device)

    ACC = victim_model.fit(
        features=features,
        adj=adj,
        labels=labels,
        idx_val=idx_val,
        idx_test=idx_test,
        beta=param,
        verbose=False,
        MI_type=args.MI_type,  # linear_CKA, DP, linear_HSIC, KDE
        stochastic=1,
        aug_pe=param['aug_pe'],
        plain_acc=param['plain_acc']
    )

    # print('=== testing GCN on original(clean) graph ===')
    # ACC = test(adj, features, labels, idx_test, victim_model)
    
    mindspore.save_checkpoint(victim_model, "best_gpb.ckpt")

    return ACC, victim_model


def GraphMI(victim_model):

    embedding = embedding_GCN(
        nfeat=features.shape[1], nhid=16,)


    mindspore.load_param_into_net(embedding, transfer_state_dict(
        victim_model.parameters_dict().items(), embedding.parameters_dict()), strict_load=False)

#     # Setup Attack Model
    Y_A = victim_model(features, adj)[0]
#     model = PGDAttack(model=victim_model, embedding=embedding,
#                       nnodes=adj.shape[0], loss_type='CE', device=device)

#     model = model.to(device)

#     model.attack(features, init_adj, labels, idx_attack,
#                  num_edges, epochs=args.epochs)
#     inference_adj = model.modified_adj.cpu()
#     print('=== calculating link inference AUC ===')
#     attack_AUC = metric(adj.numpy(), inference_adj.numpy(), idx_attack)

#     # homo_auc, homo_ap, hetero_auc, hetero_ap = adj_auc(
#     #     adj.numpy(),
#     #     inference_adj.numpy(),
#     #     labels
#     # )
#     # print('Homo AUC:{:.3f}, Homo AP:{:.3f}, Hetero AUC:{:.3f}, Hetero AP:{:.3f}'.format(
#     #     homo_auc, homo_ap, hetero_auc, hetero_ap
#     # )
#     # )
#     if args.arch != 'gat':
#         embedding.gc = deepcopy(victim_model.gc)

    embedding.set_layers(args.nlayer)
    H_A2 = embedding(features, adj)

    H_A2 = dot_product_decode(H_A2)
    Y_A2 = dot_product_decode(Y_A)

    idx = np.arange(adj.shape[0])
    auc_ha = metric(adj.numpy(), H_A2.numpy(), idx)
    print("AUC of I(A, HA) =", round(auc_ha, 3))
    auc_ya = metric(adj.numpy(), Y_A2.numpy(), idx)
    print("AUC of I(A, YA) =", round(auc_ya, 3))
    return auc_ha, auc_ya


plain_acc_maps = {
    'cora': 0.757,
    'citeseer': 0.6303,
    'polblogs': 0.8386,
    'usair': 0.4703,
    'brazil': 0.7308,
    'AIDS': 0.6682,
    'enzyme': 0.6461,
    'ogb_arxiv': 0.376,
}

param = {}

param['plain_acc'] = plain_acc_maps[args.dataset]
param['aug_pe'] = args.aug_pe
layer_MI_params = list(map(float, args.layer_MI))
layer_inter_params = list(map(float, args.layer_inter_MI))
for i in range(args.nlayer+1):
    param['layer-{}'.format(i)] = layer_MI_params[i]
    if (i+1) <= (args.nlayer):
        param['layer_inter-{}'.format(i)] = layer_inter_params[i]

# ACC, victim_model = objective(param)

victim_model = deepcopy(base_model)
param_not_load, _ = mindspore.load_param_into_net(victim_model, mindspore.load_checkpoint("best_gpb.ckpt"))

auc_ha, auc_ya = GraphMI(victim_model)

# print(f'AUC of I(A, H1A) = {auc_ha}')
# print(f'AUC of I(A, YA) = {auc_ya}')
# print(f'Acc = {round(ACC.item(), 3)}')

path = os.path.join('results/', args.dataset)
os.makedirs(path, exist_ok=True)

# torch.save(
#     victim_model.state_dict(),
#     os.path.join(path, f'{args.dataset}_{args.arch}_{args.nlayer}.pt')
# )

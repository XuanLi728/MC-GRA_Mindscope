import argparse
import mindspore
import numpy as np
import random
import baseline
from mind_dataset import Dataset
from mindspore import context
from mindspore import ops
from mindspore import Tensor
from models.mind_gcn import GCN, embedding_GCN
from topology_attack import PGDAttack
from copy import deepcopy
# from mindarmour.utils.LogUtil import set_level
import warnings

warnings.filterwarnings('ignore')
# set_level(0)

def preprocess(adj, features, labels, preprocess_adj=False, preprocess_feature=False, onehot_feature=False, sparse=False, device='cpu'):
    if preprocess_adj:
        adj_norm = normalize_adj(adj)

    if preprocess_feature:
        features = normalize_feature(features)

    labels = Tensor(labels)
    if sparse:
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        features = sparse_mx_to_torch_sparse_tensor(features)
    else:
        if onehot_feature == True:
            features = ops.eye(features.shape[0])
        else:
            features = Tensor(np.array(features.todense()))
        adj = Tensor(adj.todense())

    return adj, features, labels


def dot_product_decode(Z):
    if args.dataset == 'cora' or args.dataset == 'citeseer' or args.dataset == 'AIDS':
        Z = ops.matmul(Z, Z.t())
        adj = ops.relu(Z-ops.eye(Z.shape[0]))
        adj = ops.sigmoid(adj)

    if args.dataset == 'brazil' or args.dataset == 'usair' or args.dataset == 'polblogs':
        Z = mindquantum.utils.normalize(Z, p=2, dim=1)
        Z = ops.matmul(Z, Z.t())
        adj = ops.relu(Z-torch.eye(Z.shape[0]))

    return adj

def transfer_state_dict(pretrained_dict, model_dict):
    state_dict = {}
    for k, v in pretrained_dict:
        if k in model_dict.keys():
            state_dict[k] = v
    return state_dict

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
parser.add_argument('--nlayers', type=int, default=2,
                    help="number of layers in GCN.")
parser.add_argument('--arch', type=str,
                    choices=["gcn", "gat", "sage"], default='gcn')
parser.add_argument('--dataset', type=str, default='cora',
                    choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed', 'AIDS', 'usair', 'brazil', 'chameleon', 'ENZYME', 'squirrel', 'ogb_arxiv'], help='dataset')
# citseer pubmed
parser.add_argument('--density', type=float,
                    default=10000000.0, help='Edge density estimation')
parser.add_argument('--model', type=str, default='PGD',
                    choices=['PGD', 'min-max'], help='model variant')
parser.add_argument('--nlabel', type=float, default=1.0)
parser.add_argument('--iter', type=int, help="iterate times", default=1)
parser.add_argument('--max_eval', type=int,
                    help="max evaluation times for searching", default=100)
parser.add_argument('--log_name', type=str,
                    help="file name to save the result", default="result.txt")
parser.add_argument("--mode", type=str, default="evaluate", choices=["evaluate", "search", "baseline", "ensemble", "aux",
                                                                     "draw_violin", "notrain_test", "dev", "ensemble_search", "gaussian", "gcn_attack"])
parser.add_argument("--measure", type=str, default="HSIC",
                    choices=["HSIC", "MSELoss", "KL", "KDE", "CKA", "DP"])
parser.add_argument("--measure2", type=str, default="HSIC",
                    choices=["HSIC", "MSELoss", "KL", "KDE", "CKA", "DP"])
parser.add_argument("--nofeature", action='store_true')

parser.add_argument('--weight_aux', type=float, default=0,
                    help="the weight of auxiliary loss")
parser.add_argument('--weight_sup', type=float, default=1,
                    help="the weight of supervised loss")
parser.add_argument('--w1', type=float, default=0)
parser.add_argument('--w2', type=float, default=0)
parser.add_argument('--w3', type=float, default=0)
parser.add_argument('--w4', type=float, default=0)
parser.add_argument('--w5', type=float, default=0)
parser.add_argument('--w6', type=float, default=0)
parser.add_argument('--w7', type=float, default=0)
parser.add_argument('--w8', type=float, default=0)
parser.add_argument('--w9', type=float, default=0)
parser.add_argument('--w10', type=float, default=0)
parser.add_argument('--eps', type=float, default=0,
                    help="eps for adding noise")

parser.add_argument('--useH_A', action='store_true')
parser.add_argument('--useY_A', action='store_true')
parser.add_argument('--useY', action='store_true')
parser.add_argument('--ensemble', action='store_true')
parser.add_argument('--add_noise', action='store_true')
parser.add_argument('--defense', action='store_true')

args = parser.parse_args()

np.random.seed(args.seed)
random.seed(args.seed)
mindspore.set_seed(args.seed)

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

data = Dataset(root='./dataset', name=args.dataset, setting='GCN')
adj, features, labels, init_adj = data.adj, data.features, data.labels, data.init_adj

adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False, onehot_feature=False)
feature_adj = dot_product_decode(features)

idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

# choose the target nodes
idx_attack = np.array(random.sample(
    range(adj.shape[0]), int(adj.shape[0]*args.nlabel)))

num_edges = int(0.5 * args.density * adj.sum() /
                adj.shape[0]**2 * len(idx_attack)**2)

if args.nofeature:
    feature_adj = ops.eye(*feature_adj.size())
init_adj = Tensor(init_adj.todense(), dtype=mindspore.float32)

if args.arch == "gcn":
    victim_model = GCN(
        nfeat=features.shape[1], 
        nclass=labels.max().item() + 1, 
        nhid=16, 
        idx_train=idx_train, 
        nlayer=args.nlayers,
        dropout=0.5, 
        weight_decay=5e-4
    )

    victim_model = victim_model
    victim_model.fit(features, adj, labels, idx_val, train_iters=args.epochs)
    mindspore.save_checkpoint(victim_model, "gcn.ckpt")
    
    
# python mind_train.py --w1=0.01 --w6=10 --w7=10 --w9=10 --w10=1000 --lr=-2 --useH_A --useY_A --useY --measure=MSELoss --epochs=1
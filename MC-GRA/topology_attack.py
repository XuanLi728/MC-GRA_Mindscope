import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch
import utils
from base_attack import BaseAttack
from torch.nn import KLDivLoss
from mindspore.nn import MSELoss
from torch.nn import functional as F
from torchmetrics import AUROC
from tqdm import tqdm
from utils import *
import math
from copy import deepcopy
import mindspore
import numpy as np
from mindspore import context
from mindspore import ops
from mindspore import nn
from mindspore import Parameter
from mindspore import Tensor
from mindspore.experimental import optim
import random
import mind_utils

def metric(ori_adj, inference_adj, idx, index_delete):
    auroc = AUROC(task='binary')
    real_edge = ori_adj[idx, :][:, idx].reshape(-1).cpu()
    pred_edge = inference_adj[idx, :][:, idx].reshape(-1).cpu()
    real_edge = np.delete(real_edge, index_delete)
    pred_edge = np.delete(pred_edge, index_delete)
    return auroc(pred_edge, real_edge)


def dot_product_decode(Z):
    Z = ops.L2Normalize(axis=1)(Z)
    Z = ops.matmul(Z, Z.t())
    adj = ops.relu(Z-ops.eye(Z.shape[0]))
    return adj


def sampling_MI(prob, tau=0.5, reduction='mean'):
    prob = prob.clamp(1e-4, 1-1e-4)
    entropy1 = prob * torch.log(prob / tau)
    entropy2 = (1-prob) * torch.log((1-prob) / (1-tau))
    res = entropy1 + entropy2
    if reduction == 'none':
        return res
    elif reduction == 'mean':
        return torch.mean(res)
    elif reduction == 'sum':
        return torch.sum(res)


def Info_entropy(prob):
    prob = torch.clamp(prob, 1e-4, 1-1e-4)
    entropy = prob * torch.log2(prob)
    return -torch.mean(entropy)


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


class PGDAttack(BaseAttack):

    def __init__(self, model=None, embedding=None, H_A=None, Y_A=None, nnodes=None, loss_type='CE', feature_shape=None,
                 attack_structure=True, attack_features=False, device='cpu'):
        super(PGDAttack, self).__init__(model, nnodes,
                                        attack_structure, attack_features, device)

        assert attack_features or attack_structure, 'attack_features or attack_structure cannot be both False'

        self.loss_type = loss_type
        self.modified_adj = None
        self.modified_features = None
        self.edge_select = None
        self.complementary = None
        self.complementary_after = None
        self.embedding = embedding
        self.H_A = H_A
        self.Y_A = Y_A
        self.adj_changes_after = torch.zeros(
            int(nnodes * (nnodes - 1) / 2), requires_grad=True)
        if attack_structure:
            assert nnodes is not None, 'Please give nnodes='
            self.adj_changes = Parameter(ops.zeros(
                int(nnodes * (nnodes - 1) / 2)))

        if attack_features:
            assert True, 'Topology Attack does not support attack feature'

    def test(self, idx_attack, idx_val, idx_test, adj, features, labels, victim_model):

        victim_model.set_train(False)
        adj_norm = normalize_adj_tensor(adj)
        output = victim_model(features, adj_norm)

        acc_test = accuracy(output[idx_test], labels[idx_test])

        return acc_test.item()

    def forward_fn(self, data, label=None):
            
        ori_adj, adj, ori_features, feature_adj, args, weight_param, weight_supervised, idx_attack = data
        w1, w2, _, _, _, w6, w7, w8, w9, w10 = weight_param

        
        modified_adj = self.get_modified_adj(ori_adj)
        print("forward 1")
        modified_adj = self.adding_noise(modified_adj, args.eps)
        print("forward 2")
        adj_norm = utils.normalize_adj_tensor(modified_adj)
        
        # self.victim_model.set_train(False)
        output = self.victim_model(ori_features, adj_norm)
        # adj_tmp = ops.eye(adj_norm.shape[0]) 
        # em = self.embedding(ori_features, adj_tmp)  
        # adj_changes = self.dot_product_decode(em)
        
        

        origin_loss = self._loss(
            output[idx_attack], 
            label[idx_attack]
        ) + torch.norm(self.adj_changes, p=2) * 0.001

        # origin_loss_list.append(origin_loss.item())
        loss = weight_supervised*origin_loss


        self.embedding.set_layers(1)
        H_A1 = self.embedding(ori_features, adj)
        self.embedding.set_layers(2)
        H_A2 = self.embedding(ori_features, adj)

        Y_A = self.victim_model(ori_features, adj)


        em = self.embedding(ori_features, modified_adj-ori_adj)

        self.adj_changes_after = self.dot_product_decode(em)
        modified_adj1 = self.get_modified_adj_after(ori_adj)

        # CKA = CudaCKA(device=self.device)
        calc = MSELoss() # CKA.linear_HSIC
        calc2 = MSELoss()

        if args.measure == "MSELoss":
            calc = MSELoss()

        elif args.measure == "KL":
            calc = self.calc_kl
        # if args.measure == "KDE":
        #     calc = MutualInformation(
        #         sigma=0.4, num_bins=feature_adj.shape[0], normalize=True)

        # if args.measure == "CKA":
        #     CKA = CudaCKA(device=self.device)
        #     calc = CKA.linear_CKA

        elif args.measure == "DP":
            calc = self.dot_product

        # 10 constrains area:
        c1 = c2 = _ = _ = _ = c6 = c7 = c8 = c9 = c10 = 0
        if w1 != 0 and feature_adj.max() != feature_adj.min():
            c1 = w1 * calc(feature_adj, adj_norm)*1000 * \
                Align_Parameter_Cora["c1"]
            if args.measure == "KDE":
                loss += c1[0]
            elif args.measure == "HSIC":
                loss += -c1
            else:
                loss += c1
        if w2 != 0:
            c2 = w2 * calc(adj_norm, modified_adj1) * \
                100*Align_Parameter_Cora["c2"]
            if args.measure == "KDE":
                loss += c2[0]
            elif args.measure == "HSIC":
                loss += -c2
            else:
                loss += c2
        if w6 != 0:
            c6 = w6 * Info_entropy(adj_norm)*100*Align_Parameter_Cora["c6"]
            loss += c6
        if w7 != 0:
            c7 = w7 * Info_entropy(modified_adj1) * \
                Align_Parameter_Cora["c7"]
            loss += c7
        if w9 != 0:
            num_layers = self.embedding.nlayer
            for i in range(num_layers-1, num_layers):
                self.embedding.set_layers(i+1)
                em_cur = self.embedding(
                    ori_features, modified_adj - ori_adj)
                H_A_cur = self.embedding(ori_features, adj)
                if args.measure == "KDE":
                    calc2 = MutualInformation(
                        sigma=0.4, num_bins=H_A1.shape[1], normalize=True)
                    c9 = w9 * \
                        (calc2(H_A_cur[idx_attack], em_cur[idx_attack])[
                         0])*Align_Parameter_Cora["c9"]
                elif args.measure == "HSIC":
                    c9 = -1 * w9 * \
                        (calc(H_A_cur[idx_attack], em_cur[idx_attack])
                         )*Align_Parameter_Cora["c9"]
                else:
                    c9 = w9 * \
                        (calc(H_A_cur[idx_attack], em_cur[idx_attack])
                         )*Align_Parameter_Cora["c9"]
                loss += c9
        output2 = self.victim_model(ori_features, modified_adj)
        if w10 != 0:
            if args.measure == "KDE":
                calc2 = MutualInformation(
                    sigma=0.4, num_bins=Y_A.shape[1], normalize=True)
                c10 = w10 * calc2(Y_A[idx_attack], torch.softmax(output2[idx_attack], dim=1))[
                    0]*Align_Parameter_Cora["c10"]
            elif args.measure == "HSIC":
                c10 = -w10 * calc(Y_A[idx_attack], torch.softmax(
                    output2[idx_attack], dim=1))*Align_Parameter_Cora["c10"]
            else:
                c10 = w10 * calc(Y_A[idx_attack], torch.softmax(
                    output2[idx_attack], dim=1))*Align_Parameter_Cora["c10"]
            loss += c10

        return loss, output2
    
    def attack(self, args, index_delete, lr_ori, weight_aux, weight_supervised, weight_param, feature_adj,
               aux_adj, aux_feature, aux_num_edges, idx_train, idx_val, idx_test, adj,
               ori_features, ori_adj, labels, idx_attack, num_edges,
               dropout_rate, epochs=200, sample=False, **kwargs):
        '''
            Parameters:
            index_delete:           deleted zero edges, for metric
            lr_ori:                 learning rate
            weight:                 the weight for aux_loss1 and aux_loss2
            aux_adj:                adjancy matrix of aux graph
            aux_feature:            node feature of aux graph
            aux_num_edges:          no use yet.
            idx_train, idx_val:     index of nodes in train/val set. For testing, no use yet.
            idx_test:               index of nodes in test set. For testing.
            adj:                    adjancy matrix of origional graph
            ori_features:           node feature of origional graph
            labels:                 node labels of origional graph
            idx_attack:             index of nodes for recovery.
            num_edges:              no use in attack.
            epochs:                 epochs for recovery training.
            dropout_rate:           dropout rate in testing.
        '''
        
        

        if args.max_eval == 1:
            lr_ori = 10**args.lr
            
        self.args = args
        optimizer = optim.Adam([self.adj_changes], lr=lr_ori)
        plt.cla()
        self.victim_model = self.surrogate
        self.sparse_features = sp.issparse(ori_features)
        fake_labels = labels
        ori_adj, ori_features, labels = utils.to_tensor(
            ori_adj, ori_features, labels, device=self.device)
        
        # self.victim_model.set_train(False)
        # self.embedding.set_train(False)

        label_adj = np.load("./saved_data/"+args.dataset+".npy")
        label_adj = Tensor(label_adj)

        # lists for drawing
        acc_test_list = []
        origin_loss_list = []
        x_axis = []
        sparsity_list = []
        modified_adj = self.get_modified_adj(ori_adj)
        adj_norm = utils.normalize_adj_tensor(modified_adj)
        
        output = self.victim_model(ori_features, modified_adj)
        self.delete_eye(modified_adj)


        # for k, v in self.victim_model.parameters_dict().items():
        #     print(k, v.shape)
        
        adj_tmp = ops.eye(adj_norm.shape[0])
        em = self.embedding(ori_features, adj_tmp)
        # print("===== Overhere 3=====")
        # adj_changes = self.dot_product_decode(em)
        # print("===== Overhere 4=====")
        
        w1, w2, _, _, _, w6, w7, w8, w9, w10 = weight_param
        if args.max_eval == 1:
            w1 = args.w1
            w2 = args.w2
            w6 = args.w6
            w7 = args.w7
            w7 = args.w8
            w9 = args.w9
            w10 = args.w10
            
        print("Start Attacking")
        for t in tqdm(range(epochs)):

            if self.loss_type == 'CE':
                if sample:
                    lr = 200 / np.sqrt(t + 1)
                
                self.forward_fn([ori_adj, adj, ori_features, feature_adj, args, weight_param, weight_supervised, idx_attack],
                    labels)
                # grad_fn = mindspore.value_and_grad(
                #     self.forward_fn, None, optimizer.parameters, has_aux=True
                # )
                # print("=== inner 2 ===")
                # (loss, _), grads = grad_fn(
                #     [ori_adj, adj, ori_features, feature_adj, args, weight_param, weight_supervised, idx_attack],
                #     labels
                # )
                # print("=== inner 3 ===")
                # optimizer(grads)
                

            self.projection(num_edges)
            self.adj_changes.data.copy_(ops.clamp(
                self.adj_changes.data, min=0, max=1))

            em = self.embedding(ori_features, adj_norm)
            adj_changes = self.dot_product_decode(em)
            modified_adj = self.get_modified_adj2(
                ori_adj, adj_changes).detach()
            self.victim_model.set_train(False)
            modified_adj = self.get_modified_adj(ori_adj)
            sparsity_list.append(modified_adj.detach().cpu().mean())
            adj_norm2 = utils.normalize_adj_tensor(modified_adj)
            output2 = self.victim_model(ori_features, adj_norm2)
            cur_acc = mind_utils.accuracy(
                output2[idx_test], labels[idx_test]).item()
            acc_test_list.append(cur_acc)

            x_axis.append(t)

        em = self.embedding(ori_features, adj_norm)
        self.adj_changes.data = self.dot_product_decode(em)
        self.modified_adj = self.get_modified_adj(ori_adj).detach()

        self.embedding.set_layers(1)
        H_A1 = self.embedding(ori_features, self.modified_adj)
        self.embedding.set_layers(2)
        H_A2 = self.embedding(ori_features, self.modified_adj)
        Y_A2 = self.victim_model(ori_features, self.modified_adj)
        ori_HA = self.dot_product_decode2(self.H_A.detach())
        ori_YA = self.dot_product_decode2(self.Y_A.detach())
        H_A1 = self.dot_product_decode2(H_A1.detach())
        H_A2 = self.dot_product_decode2(H_A2.detach())
        Y_A2 = self.dot_product_decode2(Y_A2.detach())
        cur_adj = self.modified_adj + H_A1 + H_A2 + feature_adj + Y_A2
        if args.useH_A:
            cur_adj = cur_adj + ori_HA
        if args.useY_A:
            cur_adj = cur_adj + ori_YA
        if args.useY:
            cur_adj = cur_adj + label_adj

        self.modified_adj = cur_adj.detach()

        return 0, 0, 0, 0

    def _loss(self, output, labels):
        loss = 0
        if self.loss_type == "CE":
            loss = F.nll_loss(output, labels)
        # if self.loss_type == "CW":
        #     onehot = utils.tensor2onehot(labels)
        #     best_second_class = (output - 1000 * onehot).argmax(1)
        #     margin = output[np.arange(len(output)), labels] - \
        #         output[np.arange(len(output)), best_second_class]
        #     k = 0
        #     loss = -torch.clamp(margin, min=k).mean()
        return loss

    def projection(self, num_edges):
        if torch.clamp(self.adj_changes, 0, 1).sum() > num_edges:
            left = (self.adj_changes - 1).min()
            right = self.adj_changes.max()
            miu = self.bisection(left, right, num_edges, epsilon=1e-5)
            self.adj_changes.data.copy_(torch.clamp(
                self.adj_changes.data - miu, min=0, max=1))
        else:
            self.adj_changes.data.copy_(torch.clamp(
                self.adj_changes.data, min=0, max=1))

    def get_modified_adj2(self, ori_adj, adj_changes):

        if self.complementary is None:
            self.complementary = torch.ones_like(
                ori_adj) - torch.eye(self.nnodes).to(self.device)

        m = torch.zeros((self.nnodes, self.nnodes)).to(self.device)
        tril_indices = torch.tril_indices(
            row=self.nnodes, col=self.nnodes, offset=-1)
        m[tril_indices[0], tril_indices[1]] = adj_changes
        m = m + m.t()

        modified_adj = self.complementary * m + ori_adj

        return modified_adj

    def get_modified_adj(self, ori_adj):

        
        if self.complementary is None:

            self.complementary = ops.ones_like(
                ori_adj) - ops.eye(self.nnodes)
        
        
        m = ops.zeros((self.nnodes, self.nnodes))
        tril_indices = ops.tril_indices(
            row=self.nnodes, col=self.nnodes, offset=-1)

        m[tril_indices[0], tril_indices[1]] = self.adj_changes # <== problem

        m = m + m.t()
        modified_adj = self.complementary * m + ori_adj

        return modified_adj

    def get_modified_adj_after(self, ori_adj):

        if self.complementary_after is None:
            self.complementary_after = ops.ones_like(
                ori_adj) - ops.eye(self.nnodes)#.to(self.device)

        m = ops.zeros((self.nnodes, self.nnodes))#.to(self.device)
        tril_indices = ops.tril_indices(
            row=self.nnodes, col=self.nnodes, offset=-1)
        m[tril_indices[0], tril_indices[1]] = self.adj_changes_after
        m = m + m.t()

        modified_adj = self.complementary_after * m + ori_adj

        return modified_adj

    def bisection(self, a, b, num_edges, epsilon):
        def func(x):
            return torch.clamp(self.adj_changes - x, 0, 1).sum() - num_edges

        miu = a
        while ((b - a) >= epsilon):
            miu = (a + b) / 2
            # Check if middle point is root
            if (func(miu) == 0.0):
                break
            # Decide the side to repeat the steps
            if (func(miu) * func(a) < 0):
                b = miu
            else:
                a = miu
        return miu

    def dot_product_decode(self, Z):
        
        Z = ops.L2Normalize(axis=1)(Z)
        A_pred = ops.relu(ops.matmul(Z, Z.t()))
        tril_indices = ops.tril_indices(
            row=self.nnodes, col=self.nnodes, offset=-1)
        return A_pred[tril_indices[0], tril_indices[1]]

    def dot_product_decode2(self, Z):
        if self.args.dataset in ['cora', 'AIDS']:
            Z = ops.matmul(Z, Z.t())
            adj = ops.relu(Z-ops.eye(Z.shape[0]))
            adj = ops.sigmoid(adj)

        elif self.args.dataset == 'citeseer':            
            Z = F.normalize(Z, p=2, dim=1) 
            Z = torch.matmul(Z, Z.t())
            adj = torch.relu(Z-torch.eye(Z.shape[0]).to(self.device))
            adj = torch.sigmoid(adj)

        elif self.args.dataset == 'brazil':
            Z = torch.matmul(Z, Z.t())
            adj = torch.relu(Z-torch.eye(Z.shape[0]).to(self.device))

        elif self.args.dataset in ['polblogs', 'usair']:
            if self.args.dataset == 'polblogs' and \
                self.args.useH_A and self.args.useY_A and \
                self.args.useY:
                Z = torch.matmul(Z, Z.t())
                Z = F.normalize(Z, p=2, dim=1) 

            elif self.args.dataset == 'usair' and \
                self.args.useY and not self.args.useH_A \
                and not self.args.useY_A:
                Z = F.normalize(Z, p=3, dim=1)
                Z = torch.matmul(Z, Z.t())

            elif (self.args.dataset == 'usair' and \
                not self.args.useY and self.args.useH_A \
                and self.args.useY_A):
                Z = F.normalize(Z, p=2, dim=1)
                Z = torch.matmul(Z, Z.t())
            elif (self.args.dataset == 'usair' and \
                self.args.useY and self.args.useH_A \
                and not self.args.useY_A):
                Z = F.normalize(Z, p=5, dim=1)
                Z = torch.matmul(Z, Z.t())
            else:
                
                Z = torch.matmul(Z, Z.t())
                Z = F.normalize(Z, p=2, dim=1)
                
            adj = torch.relu(Z-torch.eye(Z.shape[0]).to(self.device))

        return adj

    def delete_eye(self, A):
        complementary = ops.ones_like(
            A) - ops.eye(self.nnodes)#.to(self.device)
        A = A*complementary

    def adding_noise(self, modified_adj, eps=0):
        noise = ops.randn_like(modified_adj)
        modified_adj += noise*eps
        modified_adj = ops.clamp(modified_adj, max=1, min=0)
        return modified_adj

    def dot_product(self, X, Y):
        return torch.norm(torch.matmul(Y.t(), X), p=2)

    def calc_kl(self, X, Y):
        X = F.softmax(X)
        Y = F.log_softmax(Y)
        kl = KLDivLoss(reduction="batchmean")
        return kl(Y, X)

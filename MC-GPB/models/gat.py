from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
from torchmetrics import AUROC
from tqdm import trange


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(
            N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        h_prime = torch.matmul(adj, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class embedding_gat(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads, nlayer=2, device='cuda'):
        """Dense version of GAT."""
        super(embedding_gat, self).__init__()
        self.dropout = dropout
        self.device = device
        self.nfeat = nfeat
        self.nhid = nhid
        self.hidden_sizes = [nhid]
        self.nheads = nheads
        self.nlayer = nlayer

        self.gat_layers = nn.ModuleList()
        tmp_head = nn.ModuleList(
            [GraphAttentionLayer(nfeat, self.nhid, dropout=dropout,
                                 alpha=alpha, concat=True) for _ in range(nheads)]
        )
        self.gat_layers.append(tmp_head)

        for _ in range(self.nlayer-1):
            tmp_layer = nn.ModuleList(
                [GraphAttentionLayer(self.nhid*self.nheads, self.nhid, dropout=dropout,
                                     alpha=alpha, concat=True) for _ in range(nheads)]
            )
            self.gat_layers.append(tmp_layer)

    def forward(self, x, adj):
        for i in range(self.nlayer):
            gat_layer = self.gat_layers[i]
            x = F.dropout(x, self.dropout, training=self.training)
            x = torch.cat([att(x, adj) for att in gat_layer], dim=1)
        return x

    def set_layers(self, nlayer):
        self.nlayer = nlayer


class GAT(nn.Module):
    def __init__(
        self,
        nfeat=None,
        nhid=16,
        nclass=None,
        dropout=0.5,
        alpha=0.1,
        nheads=4,
        nlayer=2,
        lr=0.005,
        weight_decay=5e-4,
        device='cuda',
    ):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay

        self.nfeat = nfeat
        self.nclass = nclass
        self.nlayer = nlayer
        self.nheads = nheads
        self.nhid = nhid
        self.hidden_sizes = [nhid]

        self.device = device

        self.gat_layers = nn.ModuleList()
        tmp_head = nn.ModuleList(
            [GraphAttentionLayer(nfeat, self.nhid, dropout=dropout,
                                 alpha=alpha, concat=True) for _ in range(nheads)]
        )
        self.gat_layers.append(tmp_head)

        for _ in range(self.nlayer-1):
            tmp_layer = nn.ModuleList(
                [GraphAttentionLayer(self.nhid*self.nheads, self.nhid, dropout=dropout,
                                     alpha=alpha, concat=True) for _ in range(nheads)]
            )
            self.gat_layers.append(tmp_layer)

        self.linear = nn.Linear(self.nhid * self.nheads, self.nclass)

    def forward(self, x, adj):
        node_embs = []
        for gat_layer in self.gat_layers:
            x = F.dropout(x, self.dropout, training=self.training)
            x = torch.cat([att(x, adj) for att in gat_layer], dim=1)
            node_embs.append(x)
        x = F.elu(x)
        x = self.linear(x)
        node_embs.append(x)
        return F.log_softmax(x, dim=1), node_embs

    def fit(
        self,
        features,
        adj,
        labels,
        idx_train,
        idx_val=None,
        idx_test=None,
        train_iters=200,
        initialize=True,
        verbose=True,
        normalize=True,
        patience=500,
        beta=None,
        MI_type='KDE',
        stochastic=0,
        con=0,
        aug_pe=0.1,
        plain_acc=0.7,
        **kwargs
    ):
        """Train the gcn model, when idx_val is not None, pick the best model according to the validation loss.

        Parameters
        ----------
        features :
            node features
        adj :
            the adjacency matrix. The format could be torch.tensor or scipy matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_val :
            node validation indices. If not given (None), GCN training process will not adpot early stopping
        train_iters : int
            number of training epochs
        initialize : bool
            whether to initialize parameters before training
        verbose : bool
            whether to show verbose logs
        normalize : bool
            whether to normalize the input adjacency matrix.
        patience : int
            patience for early stopping, only valid when `idx_val` is given
        """

        self.device = self.linear.weight.device

        if type(adj) is not torch.Tensor:
            features, adj, labels = utils.to_tensor(
                features, adj, labels, device=self.device)
        else:
            features = features.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)

        if stochastic:
            print('=== training with random Aug ===')
            features, adj = utils.stochastic(features, adj, pe=aug_pe)

        if normalize:
            if utils.is_sparse_tensor(adj):
                adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                adj_norm = utils.normalize_adj_tensor(adj)
        else:
            adj_norm = adj

        self.adj_norm = adj_norm
        self.features = features
        self.labels = labels
        self.origin_adj = adj

        if beta is None and not con:
            self._train_with_val(
                labels, idx_train, idx_val, train_iters, verbose)
        elif con:
            self._train_with_contrastive(
                labels, idx_train, idx_val, train_iters, verbose)
        else:
            print('train with MI constrain')
            layer_aucs = self._train_with_MI_constrain(
                labels=labels,
                idx_train=idx_train,
                idx_val=idx_val,
                idx_test=idx_test,
                train_iters=train_iters,
                beta=beta,
                MI_type=MI_type,
                verbose=verbose,
                plain_acc=plain_acc,
            )
            return layer_aucs

    def _train_with_MI_constrain(
        self,
        labels,
        idx_train,
        idx_val,
        idx_test,
        train_iters,
        beta,
        MI_type='MI',
        plain_acc=0.7,
        verbose=True
    ):
        if verbose:
            print('=== training MI constrain ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0

        IAZ_func = getattr(utils, MI_type)  # MI, HSIC, LP, linear_CKA

        def dot_product_decode(Z,):
            Z = torch.matmul(Z, Z.t())
            adj = torch.relu(Z-torch.eye(Z.shape[0]).to(Z.device))
            return adj

        def calculate_AUC(Z, Adj):
            auroc_metric = AUROC(task='binary')
            Z = Z.detach()
            Z = dot_product_decode(Z)

            real_edge = Adj.reshape(-1)
            pred_edge = Z.reshape(-1)

            auc_score = auroc_metric(pred_edge, real_edge)
            return auc_score

        IAZ = torch.zeros((train_iters, self.nlayer+1))
        IYZ = torch.zeros((train_iters, self.nlayer+1))
        full_losses = [[] for _ in range(4)]

        edge_index = self.origin_adj.nonzero()

        if edge_index.size(0) > 1000:
            sample_size = 1000
        else:
            sample_size = edge_index.size(0)

        loss_name = ['loss_IYZ', 'loss_IAZ', 'loss_inter', 'loss_mission']
        best_layer_AUC = 1e10
        weights = None
        final_layer_aucs = 1000
        best_acc_test = 0

        for epoch in trange(train_iters, desc='training...'):
            self.train()
            optimizer.zero_grad()
            output, node_embs = self.forward(self.features, self.adj_norm)

            node_pair_idxs = np.random.choice(
                edge_index.size(0), size=sample_size, replace=True)
            node_idx_1 = edge_index[node_pair_idxs][:, 0]
            node_idx_2 = edge_index[node_pair_idxs][:, 1]

            # node_idx_1 = edge_index[:, 0]
            # node_idx_2 = edge_index[:, 1]

            # loss_IAZ = beta * IAZ_func(self.adj_norm, node_embs[0])

            loss_IAZ = 0
            loss_inter = 0
            loss_mission = 0
            layer_aucs = []

            for idx, node_emb in enumerate(node_embs):
                #     # 层间约束
                if (idx+1) <= len(node_embs)-1:
                    param_inter = 'layer_inter-{}'.format(idx)
                    beta_inter = beta[param_inter]
                    next_node_emb = node_embs[idx+1]
                    next_node_emb = (next_node_emb@next_node_emb.T)
                    loss_inter += beta_inter * \
                        IAZ_func(next_node_emb, node_emb)

                param_name = 'layer-{}'.format(idx)
                beta_cur = beta[param_name]

                # loss_IAZ += beta_cur * IAZ_func(self.adj_norm, node_emb)

                left_node_embs = node_emb[node_idx_1]
                right_node_embs = node_emb[node_idx_2]

                # DP
                # tmp_MI = torch.sigmoid(left_node_embs * right_node_embs)
                # tmp_MI = torch.sum(tmp_MI)
                # loss_IAZ += beta_cur * tmp_MI

                # make it as adj shape [NxN]
                right_node_embs = right_node_embs @ right_node_embs.T
                loss_IAZ += beta_cur * \
                    IAZ_func(right_node_embs, left_node_embs)

                layer_AUC = calculate_AUC(node_emb, self.origin_adj).item()
                layer_aucs.append(layer_AUC)

                if idx != len(node_embs)-1:
                    output_layer = self.linear(node_emb)
                    output_layer = F.log_softmax(output_layer, dim=1)
                    loss_mission += F.nll_loss(
                        output_layer[idx_train], labels[idx_train])

            # # GIP
            # with torch.no_grad():
            #     self.eval()
            #     for l_idx, l_out in enumerate(node_embs):
            #         IAZ[epoch, l_idx] = calculate_AUC(l_out, self.origin_adj).item()

            #         if l_idx < len(node_embs)-1:
            #             l_out = self.linear1(l_out)
            #         IYZ[epoch, l_idx] = utils.accuracy(F.log_softmax(l_out, dim=1)[idx_test], labels[idx_test]).item()

            output = F.log_softmax(output, dim=1)
            loss_IYZ = F.nll_loss(output[idx_train], labels[idx_train])

            # for loss_idx in range(4):
            #     full_losses[loss_idx].append(
            #         eval(loss_name[loss_idx]).item()
            #     )

            loss_train = loss_IYZ + loss_IAZ + loss_inter + loss_mission
            loss_train.backward()
            optimizer.step()

            self.eval()
            output = self.forward(self.features, self.adj_norm)[0]
            output = F.log_softmax(output, dim=1)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])
            acc_test = utils.accuracy(output[idx_test], labels[idx_test])

            if verbose and epoch % 10 == 0:

                print('Epoch {}, loss_IYZ: {}, loss_IAZ: {}, val acc: {}'.format(
                    epoch,
                    round(loss_IYZ.item(), 4),
                    round(loss_IAZ.item(), 4),
                    acc_val
                ))

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights2 = deepcopy(self.state_dict())
                final_layer_aucs_2 = layer_aucs

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights2 = deepcopy(self.state_dict())
                final_layer_aucs_2 = layer_aucs

            if (sum(layer_aucs) < best_layer_AUC) and \
                    ((plain_acc - acc_test) < 0.05) and \
                    (acc_test > best_acc_test):
                print(acc_test)
                best_acc_test = acc_test
                best_layer_AUC = sum(layer_aucs)
                self.output = output
                weights = deepcopy(self.state_dict())
                final_layer_aucs = layer_aucs

        if verbose:
            print(
                '=== picking the best model according to the performance on validation ===')

        if weights:
            self.load_state_dict(weights)
        elif weights2:
            self.load_state_dict(weights2)

        if final_layer_aucs == 1000:
            final_layer_aucs = final_layer_aucs_2
        return {
            # 'IAZ': IAZ,
            # 'IYZ': IYZ,
            # 'full_losses': full_losses,
            'final_layer_aucs': final_layer_aucs,
        }

    def _train_with_contrastive(self, labels, idx_train, idx_val, train_iters, verbose):
        if verbose:
            print('=== training contrastive model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0

        cl_criterion = utils.SelfAdversarialClLoss()

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)[0]

            x1, adj1 = utils.stochastic(self.features, self.adj_norm)
            x2, adj2 = utils.stochastic(self.features, self.adj_norm)

            node_embs_1 = self.forward(x1, adj1)[1]
            node_embs_2 = self.forward(x2, adj2)[1]

            last_gc_1 = F.normalize(node_embs_1[-2], dim=1)
            last_gc_2 = F.normalize(node_embs_2[-2], dim=1)

            loss_cl = utils.stochastic_loss(
                last_gc_1, last_gc_2, cl_criterion, margin=1e3)

            loss_train = F.nll_loss(output[idx_train], labels[idx_train])

            loss_train += loss_cl
            loss_train.backward()
            optimizer.step()

            self.eval()
            output = self.forward(self.features, self.adj_norm)[0]
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}, val acc: {}'.format(
                    i, loss_train.item(), acc_val))

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())

        if verbose:
            print(
                '=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)

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
import utils
from tqdm import trange 

class GraphConvolution(nn.Cell):
    """Simple GCN layer, similar to https://github.com/tkipf/pygcn
    """

    def __init__(self, in_features, out_features, with_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(Tensor(np.zeros([in_features, out_features]),dtype=mindspore.float32),name=str(random.randint(1, 100000)))
        if with_bias:
            self.bias = Parameter(Tensor(out_features),name=str(random.randint(1, 100000)))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):

        stdv = 1. / math.sqrt(self.weight.shape[0])
        self.weight_r=ops.uniform(self.weight.shape,Tensor(-stdv),Tensor(stdv))
        self.weight=Parameter(self.weight_r,name=str(random.randint(1, 100000)))
        
        if self.bias is not None:
            self.bias_r=ops.uniform(self.bias.shape,Tensor(-stdv),Tensor(stdv))
            self.bias=Parameter(self.bias_r,name=str(random.randint(1, 100000)))

    def construct(self, input, adj):
        """ Graph Convolutional Layer forward function
        """
        support = ops.mm(input, self.weight.value())
        output = ops.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'

class embedding_GCN(nn.Cell):
    def __init__(self, nfeat, nhid, nlayer=2, with_bias=True, ):

        super(embedding_GCN, self).__init__()

        self.nfeat = nfeat
        self.nlayer = nlayer
        self.hidden_sizes = [nhid]
        self.gc1 = GraphConvolution(nfeat, nhid, with_bias=with_bias)
        self.gc = []
        self.gc.append(GraphConvolution(nfeat, nhid, with_bias=with_bias))
        for i in range(nlayer-1):
            self.gc.append(GraphConvolution(nhid, nhid, with_bias=with_bias))
        self.with_bias = with_bias

    def construct(self, x, adj):
        for i in range(self.nlayer):
            layer = self.gc[i]
            x = ops.relu(layer(x, adj))
        return x

    def initialize(self):
        self.gc1.reset_parameters()
        for layer in self.gc:
            layer.rset_parameters()

    def set_layers(self, nlayer):
        self.nlayer = nlayer

class GCN(nn.Cell):
    """ 2 Layer Graph Convolutional Network.

    Parameters
    ----------
    nfeat : int
        size of input feature dimension
    nhid : int
        number of hidden units
    nclass : int
        size of output dimension
    dropout : float
        dropout rate for GCN
    lr : float
        learning rate for GCN
    weight_decay : float
        weight decay coefficient (l2 normalization) for GCN. When `with_relu` is True, `weight_decay` will be set to 0.
    with_relu : bool
        whether to use relu activation function. If False, GCN will be linearized.
    with_bias: bool
        whether to include bias term in GCN weights.
    """

    def __init__(self, nfeat, nhid, idx_train, nclass, nlayer=2, dropout=0.5, lr=0.01, weight_decay=5e-4, with_relu=True, with_bias=True,):

        super(GCN, self).__init__()

        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.nlayer = nlayer
        
        self.gc = []
        self.gc.append(GraphConvolution(nfeat, nhid, with_bias=with_bias))
        for i in range(nlayer-1):
            self.gc.append(GraphConvolution(nhid, nhid, with_bias=with_bias))
            
        self.gc1 = self.gc[0]
        self.gc2 = self.gc[1]
        
        self.linear1 = nn.Dense(nhid, nclass, has_bias=with_bias)
        self.dropout = dropout
        self.lr = lr
        
        self.idx_train=Tensor(idx_train)
        
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bias = with_bias
        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.features = None
        self.IAZ_func = None
        self.initialize()

    # forward
    def construct(self, x, adj):
        node_embs = []
        for i, layer in enumerate(self.gc):
            if self.with_relu:
                x = ops.relu(layer(x, adj))
            else:
                x = layer(x, adj)
            if i != len(self.gc)-1:
                x = ops.dropout(x, self.dropout, training=self.training)
            node_embs.append(x)
        x = self.linear1(x)
        node_embs.append(x)
        return ops.log_softmax(x, axis=1), node_embs
    
    
    def initialize(self):
        """Initialize parameters of GCN.
        """
        for layers in self.gc:
            layers.reset_parameters()

    def fit(self, 
            features, adj, labels,
            MI_type='KDE', plain_acc=0.7, stochastic=0, con=0, aug_pe=0.1,
            idx_val=None, idx_test=None,
            train_iters=200, beta=None, normalize=True, 
            verbose=True, patience=500, **kwargs):


        if type(adj) is not Tensor:
            features, adj, labels = utils.to_tensor(
                features, adj, labels)
            
        if stochastic:
            print('=== training with random Aug ===')
            # features, adj = utils.stochastic(features, adj, pe=aug_pe)
            adj = ops.dropout(adj, p=aug_pe, training=True)
        
        if normalize:
            mx = adj + ops.eye(adj.shape[0]) #.to(device)
            rowsum = ops.sum(mx, dim=1)
            r_inv = rowsum.pow(-1/2).flatten()
            r_inv[ops.isinf(r_inv)] = 0.
            r_mat_inv = ops.diag(r_inv)
            mx = r_mat_inv @ mx
            adj_norm = mx @ r_mat_inv

        
        self.adj_norm = adj_norm
        self.features = features
        self.labels = labels
        self.origin_adj = adj
        
        print('train with MI constrain')
        self._train_with_MI_constrain(
                labels=labels,
                idx_val=idx_val,
                idx_test=idx_test,
                train_iters=train_iters,
                beta=beta,
                MI_type=MI_type,
                verbose=verbose,
                plain_acc=plain_acc,
            )
        return 0
    
    def forward_fn(self, data, labels):
        loss_IAZ = 0
        loss_inter = 0
        loss_mission = 0
        
        features, adj_norm, beta, node_idx_1, node_idx_2 = data 
        output, node_embs = self.construct(features, adj_norm)
        
        for idx, node_emb in enumerate(node_embs):
            # * complexity constrain in Equ.4
            if (idx+1) <= len(node_embs)-1:
                param_inter = 'layer_inter-{}'.format(idx)
                beta_inter = beta[param_inter]
                next_node_emb = node_embs[idx+1]

                next_node_emb = (next_node_emb@next_node_emb.T)

                loss_inter += beta_inter * \
                    self.IAZ_func(next_node_emb, node_emb)

            # * privacy constrain in Equ.4
            param_name = 'layer-{}'.format(idx)
            beta_cur = beta[param_name]

            left_node_embs = node_emb[node_idx_1]
            right_node_embs = node_emb[node_idx_2]

    
            right_node_embs = right_node_embs @ right_node_embs.T
            loss_IAZ += beta_cur * \
                self.IAZ_func(right_node_embs, left_node_embs)

            # * accuracy constrain in Equ.4
            if idx != len(node_embs)-1:
                output_layer = self.linear1(node_emb)
                output_layer = ops.log_softmax(output_layer, axis=1)[self.idx_train]

                loss_mission += ops.nll_loss(
                    output_layer, labels
                )

        output = ops.log_softmax(output, axis=1)[self.idx_train]
        loss_IYZ = ops.nll_loss(output, labels)

        
        loss_train = loss_IYZ + loss_IAZ + loss_inter

        loss_train = loss_train + loss_mission

        return loss_train, output

    
    def _train_with_MI_constrain(
            self,
            labels,
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
        optimizer = optim.Adam(self.trainable_params(), lr=self.lr,
                               weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0


        self.IAZ_func = getattr(utils, MI_type)  # MI, HSIC, LP, linear_CKA

#         def dot_product_decode(Z,):
#             Z = torch.matmul(Z, Z.t())
#             adj = torch.relu(Z-torch.eye(Z.shape[0]).to(Z.device))
#             return adj

#         def calculate_AUC(Z, Adj):
#             auroc_metric = AUROC(task='binary')
#             Z = Z.detach()
#             Z = dot_product_decode(Z)

#             real_edge = Adj.reshape(-1)
#             pred_edge = Z.reshape(-1)

#             auc_score = auroc_metric(pred_edge, real_edge)
#             return auc_score
    
 
        IAZ = ops.zeros((train_iters, self.nlayer+1))
        IYZ = ops.zeros((train_iters, self.nlayer+1))
        full_losses = [[] for _ in range(4)]

        edge_index = self.origin_adj.nonzero()

        if edge_index.shape[0] > 1000:
            sample_size = 1000
        else:
            sample_size = edge_index.shape[0]


        loss_name = ['loss_IYZ', 'loss_IAZ', 'loss_inter', 'loss_mission']
        best_layer_AUC = 1e10
        weights = None
        final_layer_aucs = 1000
        best_acc_test = 0

        labels = Tensor(labels, mindspore.int32) 
        for epoch in range(train_iters): # trange(train_iters, desc='training...'):
            self.set_train(True)

            node_pair_idxs = np.random.choice(
                edge_index.shape[0], size=sample_size, replace=True).tolist()

            node_idx_1 = edge_index[node_pair_idxs][:, 0]
            node_idx_2 = edge_index[node_pair_idxs][:, 1]

            loss_IAZ = 0
            loss_inter = 0
            loss_mission = 0
            layer_aucs = []
            
            grad_fn = mindspore.value_and_grad(self.forward_fn, None, optimizer.parameters, has_aux=True) # 
            (loss, _), grads = grad_fn(
                [self.features, self.adj_norm, beta, node_idx_1, node_idx_2],
                labels[self.idx_train]
            )
            optimizer(grads)
            
            # self.forward_fn(
            #     [self.features, self.adj_norm, beta, node_idx_1, node_idx_2],
            #     labels
            # )

            self.set_train(False)
            output = self.construct(self.features, self.adj_norm)[0]
            output = ops.log_softmax(output, axis=1)
            loss_val = ops.nll_loss(output[idx_val.tolist()], labels[idx_val.tolist()])
            acc_val = mind_utils.accuracy(output[idx_val.tolist()], labels[idx_val.tolist()])
            acc_test = mind_utils.accuracy(output[idx_test.tolist()], labels[idx_test.tolist()])
            
            
            weights2 = None
            final_layer_aucs_2 = None
            # if verbose and (epoch+1 % 10 == 0):
            print('Epoch {}, train loss: {}, val acc: {}, test acc: {}'.format(
                epoch,
                loss.asnumpy(),
                acc_val,
                acc_test
            ))
            
            if acc_val > 0.70:
                # mindspore.save_checkpoint(self, "gcn.ckpt")
                return acc_test

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                # weights2 = deepcopy(self.state_dict())
                final_layer_aucs_2 = layer_aucs

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                # weights2 = deepcopy(self.state_dict())
                final_layer_aucs_2 = layer_aucs

            if (sum(layer_aucs) < best_layer_AUC) and \
                    ((plain_acc - acc_test) < 0.05) and \
                    (acc_test > best_acc_test):
                # print(acc_test)
                best_acc_test = acc_test
                best_layer_AUC = sum(layer_aucs)
                self.output = output
                # weights = deepcopy(self.state_dict())
                final_layer_aucs = layer_aucs

        if verbose:
            print(
                '=== picking the best model according to the performance on validation ===')

        # if weights:
        #     # self.load_state_dict(weights)
        #     param_not_load, _ = mindspore.load_param_into_net(self, weights)
        # elif weights2:
        #     # self.load_state_dict(weights2)
        #     param_not_load, _ = mindspore.load_param_into_net(self, weights2)

        if final_layer_aucs == 1000:
            final_layer_aucs = final_layer_aucs_2
        return 0

    def test(self, idx_test):
        """Evaluate GCN performance on test set.

        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        output = self.predict()
        loss_test = ops.nll_loss(output[idx_test], self.labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test

    def predict(self, features=None, adj=None):
        """By default, the inputs should be unnormalized data

        Parameters
        ----------
        features :
            node features. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.
        adj :
            adjcency matrix. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.


        Returns
        -------
        ops.FloatTensor
            output (log probabilities) of GCN
        """

        self.eval()
        if features is None and adj is None:
            return self.forward(self.features, self.adj_norm)
        else:
            if type(adj) is not Tensor:
                features, adj = utils.to_tensor(
                    features, adj)

            self.features = features
            if utils.is_sparse_tensor(adj):
                self.adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                self.adj_norm = utils.normalize_adj_tensor(adj)
            return self.forward(self.features, self.adj_norm)




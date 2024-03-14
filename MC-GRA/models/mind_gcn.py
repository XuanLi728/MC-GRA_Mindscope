#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
# In[2]:


class GraphConvolution(nn.Cell):
    """Simple GCN layer, similar to https://github.com/tkipf/pygcn
    """

    def __init__(self, in_features, out_features, with_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        #print(in_features,out_features)
        self.weight = Parameter(Tensor(np.zeros([in_features, out_features]),dtype=mindspore.float32),name=str(random.randint(1, 100000)))
        if with_bias:
            self.bias = Parameter(Tensor(out_features),name=str(random.randint(1, 100000)))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):

        stdv = 1. / math.sqrt(self.weight.shape[0])
        #print(self.weight.value().shape)
        self.weight_r=ops.uniform(self.weight.shape,Tensor(-stdv),Tensor(stdv))
        self.weight=Parameter(self.weight_r,name=str(random.randint(1, 100000)))
        
        #self.weights.value().ops.uniform(-stdv, stdv)
        if self.bias is not None:
            self.bias_r=ops.uniform(self.bias.shape,Tensor(-stdv),Tensor(stdv))
            self.bias=Parameter(self.bias_r,name=str(random.randint(1, 100000)))

    def construct(self, input, adj):
        """ Graph Convolutional Layer forward function
        """
        #if input.data.is_sparse:
            #support = ops.spmm(input, self.weight)
        #else:
            #support = ops.mm(input, self.weight)
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



# In[3]:


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
    device: str
        'cpu' or 'cuda'.

    Examples
    --------
        We can first load dataset and then train GCN.

    >>> from deeprobust.graph.data import Dataset
    >>> from deeprobust.graph.defense import GCN
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> gcn = GCN(nfeat=features.shape[1],
              nhid=16,
              nclass=labels.max().item() + 1,
              dropout=0.5, device='cpu')
    >>> gcn = gcn.to('cpu')
    >>> gcn.fit(features, adj, labels, idx_train) # train without earlystopping
    >>> gcn.fit(features, adj, labels, idx_train, idx_val, patience=30) # train with earlystopping

    """

    def __init__(self, nfeat, nhid, nclass, idx_train,nlayer=2, dropout=0.5, lr=0.01, weight_decay=5e-4, with_relu=True, with_bias=True):

        super(GCN, self).__init__()

        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.nlayer = nlayer
        self.gc = []
        self.gc.append(GraphConvolution(nfeat, nhid, with_bias=with_bias))
        size = nhid
        last = nhid
        for i in range(nlayer-1):
            size, last = last, size
            self.gc.append(GraphConvolution(last, size, with_bias=with_bias))
        self.gc1 = self.gc[0]
        self.gc2 = self.gc[1]
        self.linear1 = nn.Dense(size, nclass, has_bias=with_bias)
        self.dropout = dropout
        self.lr = lr
        
        self.idx_train=idx_train
        
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

    def construct(self, x, adj):
        
        for i, layer in enumerate(self.gc):
            if self.with_relu:
                # print(x.shape,adj.shape)
                
                x = ops.relu(layer(x, adj))
            else:
                x = layer(x, adj)
            if i != len(self.gc)-1:
                x = ops.dropout(x, self.dropout, training=self.training)
        x = self.linear1(x)
        return ops.log_softmax(x, axis=1)
    
    
    def initialize(self):
        """Initialize parameters of GCN.
        """
        for layers in self.gc:
            layers.reset_parameters()

    def fit(self, features, adj, labels,idx_val=None, train_iters=20, initialize=True, verbose=True, normalize=True, patience=500, **kwargs):
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
        if initialize:
            self.initialize()

        if type(adj) is not Tensor:
            features, adj, labels = utils.to_tensor(
                features, adj, labels)
        else:
            features = features
            adj = adj
            labels = labels

        adj_norm = adj

        self.adj_norm = adj_norm
        self.features = features
        self.labels = labels

        if idx_val is None:
            self._train_without_val(labels, idx_train, train_iters, verbose)
        else:
            if patience < train_iters:
                self._train_with_early_stopping(
                    labels,  idx_val, train_iters, patience, verbose)
            else:
#                 Using this as training funciton
                self._train_with_val(
                    labels,idx_val, train_iters, verbose)

    def _train_without_val(self, labels,train_iters, verbose):
        optimizer = optim.Adam(self.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)

        for i in range(train_iters):
            
            output = self.forward(self.features, self.adj_norm)

            saved_var = dict()
            for tensor_name, tensor in self.named_parameters():
                saved_var[tensor_name] = ops.zeros_like(tensor)

            loss_train = ops.nll_loss(output[idx_train], labels[idx_train])
            grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
            loss_train.backward()

            ops.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            for tensor_name, tensor in self.named_parameters():
                new_grad = tensor.grad
                saved_var[tensor_name].add_(new_grad)

            for tensor_name, tensor in self.named_parameters():
                noise = Tensor(tensor.grad.shape).normal_(0, 0.05)
                saved_var[tensor_name].add_(noise)
                tensor.grad = saved_var[tensor_name]

            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

        self.eval()
        output = self.forward(self.features, self.adj_norm)
        self.output = output
           
    def _train_with_val(self,labels, idx_val, train_iters, verbose):
        
        def train_step(features,labels):
            
            grad_fn = mindspore.value_and_grad(forward_fn,None, optimizer.parameters, has_aux=True)
            (loss, _), grads = grad_fn(features,labels.astype(mindspore.int32))
            optimizer(grads)
            return loss
    
        def forward_fn(features,labels):
            logits = self.construct(features, self.adj_norm)
            loss = ops.nll_loss(logits, labels.astype(mindspore.int32))
            return loss,logits
        
        
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.trainable_params(), lr=self.lr,
                               weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0

        for i in range(train_iters):
            self.set_train(True)
            
            
            output = self.construct(self.features, self.adj_norm)      
            loss_train = train_step(self.features,labels.astype(mindspore.int32))
            self.set_train(False)
            output = self.construct(self.features, self.adj_norm)
            loss_val = ops.nll_loss(output[Tensor(idx_val)], labels[Tensor(idx_val)].astype(mindspore.int32))
            acc_val = mind_utils.accuracy(output[Tensor(idx_val)], labels[Tensor(idx_val)].astype(mindspore.int32))
            
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {} , val acc: {}'.format(i, loss_train.item(),acc_val))

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.trainable_params())

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.trainable_params())

        if verbose:
            print(
                '=== picking the best model according to the performance on validation ===')
        #self.load_param_into_net(weights)

    def _train_with_early_stopping(self, labels, idx_train, idx_val, train_iters, patience, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)

        early_stopping = patience
        best_loss_val = 100

        for i in range(train_iters):
            optimizer.clear_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = ops.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward(self.features, self.adj_norm)

            loss_val = ops.nll_loss(output[idx_val], labels[idx_val])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())
                patience = early_stopping
            else:
                patience -= 1
            if i > early_stopping and patience <= 0:
                break

        if verbose:
            print('=== early stopping at {0}, loss_val = {1} ==='.format(
                i, best_loss_val))
        self.load_param_into_net(weights)

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


# In[4]:


class embedding_GCN(nn.Cell):
    def __init__(self, nfeat, nhid, nlayer=2, with_bias=True, ):

        super(embedding_GCN, self).__init__()

        self.nfeat = nfeat
        self.nlayer = nlayer
        self.hidden_sizes = [nhid]
        self.emb_gc1 = GraphConvolution(nfeat, nhid, with_bias=with_bias)
        self.emb_gc = []
        self.emb_gc.append(GraphConvolution(nfeat, nhid, with_bias=with_bias))
        for i in range(nlayer-1):
            self.emb_gc.append(GraphConvolution(nhid, nhid, with_bias=with_bias))
        self.with_bias = with_bias

    def construct(self, x, adj):
        for i in range(self.nlayer):

            layer = self.emb_gc[i]
            x = ops.relu(layer(x, adj))
        return x

    def initialize(self):
        self.emb_gc1.reset_parameters()
        for layer in self.emb_gc:
            layer.rset_parameters()

    def set_layers(self, nlayer):
        self.nlayer = nlayer





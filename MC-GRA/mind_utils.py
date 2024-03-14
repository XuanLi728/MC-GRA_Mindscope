import math
import random
import numpy as np
import scipy.sparse as sp
import mindspore
from mindspore import nn
from mindspore import ops

def accuracy(output, labels):
    """Return accuracy of output compared to labels.

    Parameters
    ----------
    output : torch.Tensor
        output from model
    labels : torch.Tensor or numpy.array
        node labels

    Returns
    -------
    float
        accuracy
    """
    if labels is int:
        labels = [labels]
    if type(labels) is not ops.Tensor:
        labels = ops.Tensor(labels)
    values,preds = output.max(axis=1,return_indices=True)
    #print(preds)
    #print(labels)
    #print(output.shape)
    correct=ops.equal(preds,labels)
    #print(correct)
    correct = correct.sum().astype(mindspore.float32)
    #print(correct)
    return correct / len(labels)

'''def accuracy(output, labels):
    """Return accuracy of output compared to labels.

    Parameters
    ----------
    output : torch.Tensor
        output from model
    labels : torch.Tensor or numpy.array
        node labels

    Returns
    -------
    float
        accuracy
    """
    if labels is int:
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)'''

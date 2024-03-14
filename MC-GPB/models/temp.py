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
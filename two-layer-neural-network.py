
import torch
import random
import statistics
# from linear-classifier import sample_batch
from typing import Dict, List, Callable, Optional



# Template class modules that we will use later: Do not edit/modify this class
class TwoLayerNet(object):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dtype: torch.dtype = torch.float32,
        device: str = "cuda",
        std: float = 1e-4,
    ):
        
        # reset seed before start
        random.seed(0)
        torch.manual_seed(0)

        self.params = {}
        self.params["W1"] = std * torch.randn(
            input_size, hidden_size, dtype=dtype, device=device
        )
        self.params["b1"] = torch.zeros(hidden_size, dtype=dtype, device=device)
        self.params["W2"] = std * torch.randn(
            hidden_size, output_size, dtype=dtype, device=device
        )
        self.params["b2"] = torch.zeros(output_size, dtype=dtype, device=device)

    def loss(
        self,
        X: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        reg: float = 0.0,
    ):
        return nn_forward_backward(self.params, X, y, reg)

    def train(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        learning_rate: float = 1e-3,
        learning_rate_decay: float = 0.95,
        reg: float = 5e-6,
        num_iters: int = 100,
        batch_size: int = 200,
        verbose: bool = False,
    ):
        # fmt: off
        return nn_train(
            self.params, nn_forward_backward, nn_predict, X, y,
            X_val, y_val, learning_rate, learning_rate_decay,
            reg, num_iters, batch_size, verbose,
        )
        # fmt: on

    def predict(self, X: torch.Tensor):
        return nn_predict(self.params, nn_forward_backward, X)

    def save(self, path: str):
        torch.save(self.params, path)
        print("Saved in {}".format(path))

    def load(self, path: str):
        checkpoint = torch.load(path, map_location="cpu")
        self.params = checkpoint
        if len(self.params) != 4:
            raise Exception("Failed to load your checkpoint")

        for param in ["W1", "b1", "W2", "b2"]:
            if param not in self.params:
                raise Exception("Failed to load your checkpoint")
        # print("load checkpoint file: {}".format(path))


def nn_forward_pass(params: Dict[str, torch.Tensor], X: torch.Tensor):
    
    # Unpack variables from the params dictionary
    W1, b1 = params["W1"], params["b1"]
    W2, b2 = params["W2"], params["b2"]
    N, D = X.shape

    # Compute the forward pass
    hidden = None
    scores = None

    # compute the x * W1 with hidden layer (layer1)
    hidden = X.mm(W1) + b1
    # relu activation function sets all -ve numbers to zero
    hidden[hidden < 0] = 0
    # layer 2 - output scores 
    scores = hidden.mm(W2) + b2

    return scores, hidden


def nn_forward_backward(
    params: Dict[str, torch.Tensor],
    X: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    reg: float = 0.0
):
    
    # Unpack variables from the params dictionary
    W1, b1 = params["W1"], params["b1"]
    W2, b2 = params["W2"], params["b2"]
    N, D = X.shape

    scores, h1 = nn_forward_pass(params, X)
    # If the targets are not given then jump out, we're done
    if y is None:
        return scores

    # Compute the loss
    loss = None
    

    num_train = X.shape[0]
    # computing softmax loss on score
    score_exp = torch.exp(scores)
    softmax_pi = score_exp / torch.sum(score_exp, dim=1, keepdim=True)
    # softmax loss of all correct classes
    softmax_li = -torch.log(softmax_pi[range(num_train), y])
    data_loss = torch.sum(softmax_li) / num_train
    # computing regularization 
    l2_reg_loss = reg * (torch.sum(W1 * W1) + torch.sum(W2 * W2))
    # total loss = regularization + data loss
    loss = data_loss + l2_reg_loss

    # Backward pass: compute gradients
    grads = {}
    
    grad_loss = softmax_pi 
    # subtract 1 from loss
    grad_loss[range(num_train), y] -= 1
    grad_loss /= num_train

    #backward pass on W2
    grads['W2'] = h1.t().mm(grad_loss)
    grads['b2'] = torch.sum(grad_loss, dim=0)
    # backward pass on hidden layer
    dhidden = grad_loss.mm(W2.t())
    # reLU
    dhidden[h1 <= 0] = 0
    # backward pass on input layer 1
    grads['W1'] = X.t().mm(dhidden)
    grads['b1'] = torch.sum(dhidden, dim=0)
    #computing regularization
    grads['W2'] += 2 * reg * W2
    grads['W1'] += 2 * reg * W1

    return loss, grads


def nn_train(
    params: Dict[str, torch.Tensor],
    loss_func: Callable,
    pred_func: Callable,
    X: torch.Tensor,
    y: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    learning_rate: float = 1e-3,
    learning_rate_decay: float = 0.95,
    reg: float = 5e-6,
    num_iters: int = 100,
    batch_size: int = 200,
    verbose: bool = False,
):
   
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train // batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []
    X_batch = []
    y_batch=[]

    for it in range(num_iters):
        # X_batch, y_batch = sample_batch(X, y, num_train, batch_size)

        # Compute loss and gradients using the current minibatch
        loss, grads = loss_func(params, X_batch, y=y_batch, reg=reg)
        loss_history.append(loss.item())

        

        params['W1'] -= grads['W1'] * learning_rate
        params['W2'] -= grads['W2'] * learning_rate
        params['b1'] -= grads['b1'] * learning_rate
        params['b2'] -= grads['b2'] * learning_rate


        if verbose and it % 100 == 0:
            print("iteration %d / %d: loss %f" % (it, num_iters, loss.item()))

        # Every epoch, check train and val accuracy and decay learning rate.
        if it % iterations_per_epoch == 0:
            # Check accuracy
            y_train_pred = pred_func(params, loss_func, X_batch)
            train_acc = (y_train_pred == y_batch).float().mean().item()
            y_val_pred = pred_func(params, loss_func, X_val)
            val_acc = (y_val_pred == y_val).float().mean().item()
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)

            # Decay learning rate
            learning_rate *= learning_rate_decay

    return {
        "loss_history": loss_history,
        "train_acc_history": train_acc_history,
        "val_acc_history": val_acc_history,
    }


def nn_predict(
    params: Dict[str, torch.Tensor], loss_func: Callable, X: torch.Tensor
):
    
    y_pred = None

    # W1, b1 = params['W1'], params['b1']
    # W2, b2 = params['W2'], params['b2']
    # #forward pass
    # hidden = X.mm(W1) + b1
    # # relu activation function
    # hidden[hidden < 0] = 0
    # # layer 2 - output scores 
    # scores = hidden.mm(W2) + b2

    # # gives highest score
    # y_pred = torch.argmax(scores, dim=1)

    scores, _ = nn_forward_pass(params, X)
    _, y_pred = scores.max(dim=1)
    return y_pred


def nn_get_search_params():
    
    learning_rates = []
    hidden_sizes = []
    regularization_strengths = []
    learning_rate_decays = []

    # learning_rates = [1e-3, 5e3, 1, 1e2]
    # hidden_sizes = [32, 64, 128, 150]
    # regularization_strengths = [1e-4,1e-7 ,0, 1e-5, 1e-3]
    #learning_rate_decays = [0.95, 1, 0]
    hidden_sizes = [8, 16, 32, 128] 
    regularization_strengths = [0, 1e-5, 1e-3, 1e0]
    learning_rates = [1e4, 1e-2, 1e0, 1e2, 1e-1]
    
    return (
        learning_rates,
        hidden_sizes,
        regularization_strengths,
        learning_rate_decays,
    )


def find_best_net(
    data_dict: Dict[str, torch.Tensor], get_param_set_fn: Callable
):
    
    best_net = None
    best_stat = None
    best_val_acc = 0.0

    X_train, y_train = data_dict['X_train'], data_dict['y_train']
    X_val, y_val = data_dict['X_val'], data_dict['y_val']

    learning_rates, hidden_sizes, regularization_strengths, learning_rate_decays = get_param_set_fn()

    # grid search
    for lr in learning_rates:
      for hs in hidden_sizes:
        for reg in regularization_strengths:
          
            two_layer_nn = TwoLayerNet(3*32*32, hs, 10)
            nn_train = two_layer_nn.train(X_train, y_train, X_val, y_val, num_iters=1000, batch_size=200, learning_rate=lr, reg=reg, learning_rate_decay=0.95, verbose=False)
            # Evaluate accuracy on validation set
            val_acc = (two_layer_nn.predict(X_val) == y_val).float().mean().item()

            # Update best model if current model has higher validation accuracy
            if val_acc > best_val_acc:
              best_net = two_layer_nn
              best_stat = nn_train
              best_val_acc = val_acc


    return best_net, best_stat, best_val_acc


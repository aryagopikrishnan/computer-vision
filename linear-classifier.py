import torch
import random
import statistics
from abc import abstractmethod
from typing import Dict, List, Callable, Optional




class LinearClassifier:
    
    def __init__(self):
        random.seed(0)
        torch.manual_seed(0)
        self.W = None

    def train(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        learning_rate: float = 1e-3,
        reg: float = 1e-5,
        num_iters: int = 100,
        batch_size: int = 200,
        verbose: bool = False,
    ):
        train_args = (
            self.loss,
            self.W,
            X_train,
            y_train,
            learning_rate,
            reg,
            num_iters,
            batch_size,
            verbose,
        )
        self.W, loss_history = train_linear_classifier(*train_args)
        return loss_history

    def predict(self, X: torch.Tensor):
        return predict_linear_classifier(self.W, X)

    @abstractmethod
    def loss(
        self,
        W: torch.Tensor,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
        reg: float,
    ):
        
        raise NotImplementedError

    def _loss(self, X_batch: torch.Tensor, y_batch: torch.Tensor, reg: float):
        self.loss(self.W, X_batch, y_batch, reg)

    def save(self, path: str):
        torch.save({"W": self.W}, path)
        print("Saved in {}".format(path))

    def load(self, path: str):
        W_dict = torch.load(path, map_location="cpu")
        self.W = W_dict["W"]
        if self.W is None:
            raise Exception("Failed to load your checkpoint")
        # print("load checkpoint file: {}".format(path))


class LinearSVM(LinearClassifier):
    """A subclass that uses the Multiclass SVM loss function"""

    def loss(
        self,
        W: torch.Tensor,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
        reg: float,
    ):
        return svm_loss_vectorized(W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
    """A subclass that uses the Softmax + Cross-entropy loss function"""

    def loss(
        self,
        W: torch.Tensor,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
        reg: float,
    ):
        return softmax_loss_vectorized(W, X_batch, y_batch, reg)


# SVM Loss


def svm_loss_naive(
    W: torch.Tensor, X: torch.Tensor, y: torch.Tensor, reg: float
):
    
    dW = torch.zeros_like(W)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = W.t().mv(X[i])
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                
                dW[:, j] += X[i]
                dW[:, y[i]] -= X[i]
                
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * torch.sum(W * W)

    
    dW /= num_train
    dW += 2 * reg * W
    
    return loss, dW


def svm_loss_vectorized(
    W: torch.Tensor, X: torch.Tensor, y: torch.Tensor, reg: float
):
    
    loss = 0.0
    dW = torch.zeros_like(W)  # initialize the gradient as zero

    
    num_train = X.shape[0]
    num_classes = W.shape[1]
    scores = X.mm(W)
    correct_class_score = scores[torch.arange(num_train), y]
    # calculating hinge loss
    margin = scores - correct_class_score.view(-1,1) + 1
    margin[torch.arange(num_train), y] = 0
    loss = margin.sum() / num_train
    loss += reg * torch.sum(W*W)
  
    
    margin[margin > 0] = 1
    margin_count = margin.sum(axis=1)
    margin[torch.arange(num_train), y] = -margin_count
    dW = (X.T).mm(margin) / num_train
    dW += 2*reg*W
    
    return loss, dW


def sample_batch(
    X: torch.Tensor, y: torch.Tensor, num_train: int, batch_size: int
):
    
    X_batch = None
    y_batch = None
   
    indices = torch.randint(num_train, (batch_size,))
    X_batch = X[indices]
    y_batch = y[indices]
    
    return X_batch, y_batch


def train_linear_classifier(
    loss_func: Callable,
    W: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    learning_rate: float = 1e-3,
    reg: float = 1e-5,
    num_iters: int = 100,
    batch_size: int = 200,
    verbose: bool = False,
):
    # assume y takes values 0...K-1 where K is number of classes
    num_train, dim = X.shape
    if W is None:
        # lazily initialize W
        num_classes = torch.max(y) + 1
        W = 0.000001 * torch.randn(
            dim, num_classes, device=X.device, dtype=X.dtype
        )
    else:
        num_classes = W.shape[1]

    # Run stochastic gradient descent to optimize W
    loss_history = []
    for it in range(num_iters):
        # TODO: implement sample_batch function
        X_batch, y_batch = sample_batch(X, y, num_train, batch_size)

        # evaluate loss and gradient
        loss, grad = loss_func(W, X_batch, y_batch, reg)
        loss_history.append(loss.item())

       
        W -= learning_rate * grad
        
        if verbose and it % 100 == 0:
            print("iteration %d / %d: loss %f" % (it, num_iters, loss))

    return W, loss_history


def predict_linear_classifier(W: torch.Tensor, X: torch.Tensor):
    
    y_pred = torch.zeros(X.shape[0], dtype=torch.int64)
    
    predicted_class = X.mm(W)
    y_pred = torch.argmax(predicted_class, dim=1)
    
    return y_pred


def svm_get_search_params():
    """
    Return candidate hyperparameters for the SVM model. You should provide
    at least two param for each, and total grid search combinations
    should be less than 25.

    Returns:
    - learning_rates: learning rate candidates, e.g. [1e-3, 1e-2, ...]
    - regularization_strengths: regularization strengths candidates
                                e.g. [1e0, 1e1, ...]
    """

    learning_rates = []
    regularization_strengths = []

   
    learning_rates = [1e-3, 1e-2, 1e-1, 1e2, 1e3]
    regularization_strengths = [1e0, 1e1, 1e-3, 1e-1, 1e-2]

    return learning_rates, regularization_strengths


def test_one_param_set(
    cls: LinearClassifier,
    data_dict: Dict[str, torch.Tensor],
    lr: float,
    reg: float,
    num_iters: int = 2000,
):
    
    train_acc = 0.0  # The accuracy is simply the fraction of data points
    val_acc = 0.0  # that are correctly classified.
    
    X_train = data_dict['X_train']
    y_train = data_dict['y_train']
    X_val = data_dict['X_val']
    y_val = data_dict['y_val']
    # loss = cls.train(X_train, y_train, learning_rate=lr, reg=reg, num_iters=100, verbose = False)
    # y_train_predicted = cls.predict(X_train)
    # # average
    # train_acc = torch.mean((y_train_predicted == y_train).float())
    # # validation set
    # y_val_predicted = cls.predict(X_val)
    # val_acc = torch.mean((y_val_predicted == y_val).float())

    cls.train(X_train, y_train, lr, reg, num_iters, batch_size=200, verbose=False)

    y_train_predicted = cls.predict(X_train)
    train_acc = 100.0 * (y_train == y_train_predicted).double().mean().item()

    y_test_predicted = cls.predict(X_val)
    val_acc = 100.0 * (y_val == y_test_predicted).double().mean().item()

    return cls, train_acc, val_acc


# Softmax 

def softmax_loss_naive(
    W: torch.Tensor, X: torch.Tensor, y: torch.Tensor, reg: float
):

    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = torch.zeros_like(W)

    num_classes = W.shape[1]
    num_train = X.shape[0]

    #compute scores and store correct class score
    scores = W.t().mv(X[0])
    correct_class_score = scores[y[0]]
    sum_of_scores = 0.0
    #softmax in for loop
    for i in range(num_train):
      scores = W.t().mv(X[i])
      scores = scores - scores.max()
      correct_class_score = scores[y[i]]
      # loss of softmax formula: Li = -log(pyi)
      loss += torch.log(torch.exp(scores).sum()) - correct_class_score
      for j in range(num_classes):
        if j ==y[i]:
          sum_of_scores =  torch.exp(scores).sum()
          dW[:, j] += torch.exp(scores[j]) / sum_of_scores * X[i, :] - X[i, :]
        else:
          dW[:, j] += torch.exp(scores[j]) / sum_of_scores * X[i, :]


      # computing loss and grad
      # adding the regularization - prevent overfitting
      loss /= num_train
      loss += reg * torch.sum(W*W)
      dW /= num_train
      dW += 2 * reg * W

    return loss, dW


def softmax_loss_vectorized(
    W: torch.Tensor, X: torch.Tensor, y: torch.Tensor, reg: float
):
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = torch.zeros_like(W)

    
    num_classes = W.shape[1]
    num_train = X.shape[0]
    # mm of X and weights - NxC matrix
    scores = X.mm(W)
    
    max_score, _  = scores.max(dim = 1)
    scores = scores - max_score.view(-1,1)
    score_exp = scores.exp()
    score_exp_sum = score_exp.sum()
    score_exp_sum_log = score_exp_sum.log()
    correct_class_score = scores[range(num_train), y]
    loss = (score_exp_sum_log - correct_class_score).sum()
    dW_zero = torch.zeros(num_train, num_classes, dtype=torch.float64, device = 'cuda')
    dW_zero[range(num_train), y] = -1
    dW += (dW_zero.t().mm(X)).t()
    dW += ((score_exp/score_exp_sum.view(-1,1)).t().mm(X)).t()

    loss /= num_train
    loss += reg * torch.sum(W*W)
    dW /= num_train
    dW += 2 * reg * W
   
    return loss, dW


def softmax_get_search_params():
    
    learning_rates = []
    regularization_strengths = []

    # learning_rates = [1, 1e-02, 1e-05, 1e-04]
    learning_rates = [10**-i for i in range(5)]
    regularization_strengths = [10**-i for i in range(4)]
    

    return learning_rates, regularization_strengths


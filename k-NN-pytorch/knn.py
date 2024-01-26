"""
Implements a K-Nearest Neighbor classifier in PyTorch.
"""
import torch
from typing import Dict, List


def hello():
    print("Hello from knn.py!")


def compute_distances_two_loops(x_train: torch.Tensor, x_test: torch.Tensor):
   
    # Initialize dists to be a tensor of shape (num_train, num_test) with the
    # same datatype and device as x_train
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    dists = x_train.new_zeros(num_train, num_test)
    train = x_train.flatten(1)
    test = x_test.flatten(1)

    for i in range(num_test):
      for j in range(num_train):
        dists[j,i] = torch.sqrt(torch.sum(torch.square(train[j] - test[i])))
    return dists


def compute_distances_one_loop(x_train: torch.Tensor, x_test: torch.Tensor):
    
    # Initialize dists to be a tensor of shape (num_train, num_test) with the
    # same datatype and device as x_train
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    dists = x_train.new_zeros(num_train, num_test)
    
    # train = x_train.flatten(1)
    # test = x_test.flatten(1)
    x_train = x_train.reshape(num_train, -1)
    x_test = x_test.reshape(num_test, -1)

    for i in range(num_train):
      dists[i] = torch.sqrt(torch.sum((x_test - x_train[i])**2, dim=1).t())
    return dists


def compute_distances_no_loops(x_train: torch.Tensor, x_test: torch.Tensor):
    
    # Initialize dists to be a tensor of shape (num_train, num_test) with the
    # same datatype and device as x_train
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    dists = x_train.new_zeros(num_train, num_test)
    
    train = x_train.flatten(1)
    test = x_test.flatten(1)
    # x^2 - y^2 = x^2 + y^2 - 2xy
    train_square = torch.square(train)
    test_square = torch.square(test)
    # expand the square in euclidian dist formula 2xy
    train_square_sum = torch.sum(train_square, 1)
    test_square_sum = torch.sum(test_square, 1)
    matrix_mult = torch.matmul(train, test.transpose(0,1))
    #sqrt to the inputs 
    dists = torch.sqrt(train_square_sum.reshape(-1,1) + test_square_sum.reshape(1,-1) - (2*matrix_mult))
    
    return dists


def predict_labels(dists: torch.Tensor, y_train: torch.Tensor, k: int = 1):
    
    num_train, num_test = dists.shape
    y_pred = torch.zeros(num_test, dtype=torch.int64)
    
    for i in range(num_test):
      # index of k lowest value
      idx_values = torch.topk(dists[:, i], k,largest=False).indices
      # store the corresponding labels from training set
      k_labels = y_train[idx_values]
      # find the most occuring label from k_label (majority vote)
      y_pred[i] = torch.argmax(torch.bincount(k_labels))

    return y_pred


class KnnClassifier:

    def __init__(self, x_train: torch.Tensor, y_train: torch.Tensor):

        self.x_train = x_train
        self.y_train = y_train
        
    def predict(self, x_test: torch.Tensor, k: int = 1):
        
        y_test_pred = None
        
        dists = compute_distances_no_loops(self.x_train, x_test)
        y_test_pred = predict_labels(dists, self.y_train, k)
        
        return y_test_pred

    def check_accuracy(
        self,
        x_test: torch.Tensor,
        y_test: torch.Tensor,
        k: int = 1,
        quiet: bool = False
    ):
        
        y_test_pred = self.predict(x_test, k=k)
        num_samples = x_test.shape[0]
        num_correct = (y_test == y_test_pred).sum().item()
        accuracy = 100.0 * num_correct / num_samples
        msg = (
            f"Got {num_correct} / {num_samples} correct; "
            f"accuracy is {accuracy:.2f}%"
        )
        if not quiet:
            print(msg)
        return accuracy


def knn_cross_validate(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    num_folds: int = 5,
    k_choices: List[int] = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100],
):
    
    # First we divide the training data into num_folds equally-sized folds.
    x_train_folds = []
    y_train_folds = []
    
    x_train_folds = torch.chunk(x_train, num_folds)
    y_train_folds = torch.chunk(y_train, num_folds)

    
    k_to_accuracies = {}

    
    for k in k_choices:
      k_to_accuracies[k] = []
      for i in range(num_folds):
        for j in range(num_folds):
          if i != j:
            fx_train = torch.cat([x_train_folds[j]])
            fy_train = torch.cat([y_train_folds[j]])
        fx_test = x_train_folds[i]
        fy_test = y_train_folds[i]

        algorithm_test = KnnClassifier(fx_train, fy_train)
        accuracy = algorithm_test.check_accuracy(fx_test, fy_test, k=k, quiet=True)
        k_to_accuracies[k].append(accuracy)


    
    return k_to_accuracies


def knn_get_best_k(k_to_accuracies: Dict[int, List]):
   
    best_k = 0
    
    import statistics
    mean_accuracy = []
    for k, accuracy in k_to_accuracies.items():
      mean_val = statistics.mean(accuracy)
      mean_accuracy.append(mean_val)
    #print(mean_accuracy)
      
    a = torch.tensor(mean_accuracy)
    sorted_items = sorted(k_to_accuracies.items())
    best_k = int(torch.argmax(a))

    return best_k

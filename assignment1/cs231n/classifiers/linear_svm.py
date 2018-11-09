import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

   # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    number = 0
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        number+=1
        dW[:,j] += np.transpose(X[i])

    dW[:,y[i]]+= -number* np.transpose(X[i])
        

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  # Add regularization to the loss.
  dW += reg * W
  loss += reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W)
  y_temp = y.copy()
  y= y.reshape(X.shape[0],1)
  y = np.transpose(y).tolist()
  correct_scores = np.transpose(scores[np.arange(scores.shape[0]),y])
  result = scores - correct_scores + 1
  result[np.arange(scores.shape[0]),y] = 0
  result[result < 0]=0
  loss = np.sum(result)
  loss/= (X.shape[0])
  loss += reg * np.sum(W*W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  num_train = X.shape[0]
  mask = result.copy()
  mask[mask>0]=1
  num_train = X.shape[0]
  mask = result.copy()
  mask[mask>0]=1
  mask[np.arange(num_train), y] = -np.sum(mask, axis=1)
  dW = X.T.dot(mask)
  dW /= num_train
  dW += 2 * reg * W

 """
  for j in range (num_train):
        number = 0
        X_temp = X[j].copy()
        X_temp = X_temp.reshape((X[j].shape[0]),1)
        temp =X_temp.dot(np.transpose(mask[j][:].reshape(W.shape[1],1)))
        dW[:,:] += temp
        dW[:,y_temp[j]] -=  X[j]
        number = np.sum(mask[j,:])-1
        dW[:,y_temp[j]]+= -1*number* X[j]

  dW /= num_train
  # Add regularization to the loss.
  dW += 2 * reg * W 
  """
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW

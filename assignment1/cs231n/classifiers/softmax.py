import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  
  N, D = X.shape
  C = W.shape[1]
  S = np.zeros((N,C))
  dS = np.zeros((N,C))
  
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  
   # forward
  for i in range(N):                      # for each training sample
    for j in range(C):                    # for each class
      for k in range(D):
          S[i, j] += X[i, k] * W[k, j]    #Multiplication of vectors
    
    S[i, :] -= np.max(S[i, :])             
    S[i, :] = np.exp(S[i, :])             #exponential
    S[i, :] /= np.sum(S[i, :])            #Normalization
  
  
    
  
  
  for i in range(N):                       # for each training sample
    for j in range(C):                     # for each class
        if j == y[i]:                      # when the index is the "correct class" index for the sample
            
            loss+= -np.log(S[i,j])           #loss = sum(loss of every example)
            
            dS[i,j] = (-1/N) * (1-S[i,j])   #if label is correct ds = -1/N*(1-s) 
        else:                              # every other index
            dS[i,j] = (1.0/N) * ( S[i,j] )    #ds = (1/N)*(s)
  
    # compute loss
  loss /= N
  loss +=   reg * np.sum(W**2) 

   #gradient 
  dW = np.dot(X.T, dS)
  dW += 0.5 * reg * W    
    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  N = X.shape[0]
  ds =  np.zeros_like(W)
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
 
  s= np.dot(X, W) # (N, C)              #S = W.X
  
  #compute loss
  s -= np.max(s)                         #to avoid numeric stability
  s= np.exp(s)                           #exponential
  s /= np.sum(s,axis=1,keepdims=True)    #probalbility 
  loss -= (1/N)*np.sum(np.log(s[np.arange(N), y]))     #get loss = 1/N(sigma(log(k)))where K is when S is the correct class 
  loss +=  reg * np.sum(W**2)                          #loss = loss + R(W)
  
  #gradient
  ds = (1/N)*s              #ds = (1/N)*(s)
  
  ds[np.arange(N), y] -= (1/N)    #if label is correct ds = -1/N*(1-s) = (1/N)*(s)- (1/N)
  
  dW = np.dot(X.T, ds)
  
  dW +=(1/2) * reg * W 
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


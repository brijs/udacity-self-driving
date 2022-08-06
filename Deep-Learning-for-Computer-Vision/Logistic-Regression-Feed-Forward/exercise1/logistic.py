import tensorflow as tf

from utils import check_softmax, check_ce, check_model, check_acc

def softmax(logits):
    """
    softmax implementation
    args:
    - logits [tensor]: 1xN logits tensor
    returns:
    - soft_logits [tensor]: softmax of logits
    """
    # IMPLEMENT THIS FUNCTION
    # softmax(x) = e^x / sum(e^x); x: logit sscores
    n = tf.math.exp(logits)
    d = tf.math.reduce_sum(n,1,keepdims=True)
    soft_logits = n/d
    
    return soft_logits


def cross_entropy(scaled_logits, one_hot):
    """
    Cross entropy loss implementation
    args:
    - scaled_logits [tensor]: NxC tensor where N batch size / C number of classes
    - one_hot [tensor]: one hot tensor
    returns:
    - loss [tensor]: cross entropy 
    """
    # IMPLEMENT THIS FUNCTION
    # logits: NxC; one_hot: Cx1 => Nx1
    # "scaled logits" means these are normalized using softmax; ie it is now a probability(not raw scores)
    
    # this computes the sum of product; essentially it's just 1 term since there is one 1
    temp = tf.boolean_mask(scaled_logits, one_hot)
    nll = -tf.math.log(temp) # The log here is applied after summation but again it doesn't matter since there is only one 1
    return nll


def model(X, W, b):
    """
    logistic regression model
    args:
    - X [tensor]: input HxWx3
    - W [tensor]: weights
    - b [tensor]: bias
    returns:
    - output [tensor]
    """
    # IMPLEMENT THIS FUNCTION
#     (28, 28, 3) (2352, 10) (10,) => inputs = 28*28*3; Ouputs = 10
#     print(X.shape, W.shape, b.shape)
    flat_X = tf.reshape(X, (-1, W.shape[0]) )
    return softmax(tf.matmul(flat_X, W) + b)


def accuracy(y_hat, Y):
    """
    calculate accuracy
    args:
    - y_hat [tensor]: NxC tensor of models predictions
    - y [tensor]: N tensor of ground truth classes
    returns:
    - acc [tensor]: accuracy
    """
    # IMPLEMENT THIS FUNCTION
    #     (2, 5) (2,)
    # there are N=2 observations; C=5 output classes
#     print (y_hat.shape, Y.shape)

    # accuracy is simply what percentage of predictions(y_hat rows) are correct; 
    # to determine the model's prediction, we'll pick the max in y_hat. The index will correspond to the class number
    predicted_classes = tf.cast(tf.argmax(y_hat, axis=1), Y.dtype)
    num_predictions = Y.shape[0]
    acc = tf.math.reduce_sum(tf.cast(predicted_classes == Y, tf.int32)) / num_predictions
    return acc


if __name__ == '__main__':
    check_softmax(softmax)
    check_ce(cross_entropy)
    check_model(model)
    check_acc(accuracy)

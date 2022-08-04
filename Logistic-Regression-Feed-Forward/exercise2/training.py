import sys
import logging
import argparse

import tensorflow as tf

from dataset import get_datasets
from logistic import softmax, cross_entropy, accuracy


def get_module_logger(mod_name):
    logger = logging.getLogger(mod_name)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger


def model(W, X, b):
    """
    args:
    - W: (1024*3, 43) | (input_shape, num_output_classes)
    - X: (256,32,32,3) | (batch_size, image_shape) | 256 rows (images)
    -       => flatten to (256, 1024 * 3)
    - b: (43,) | 43 output classes
    returns:
    - WX + b: (256,) | 1-D output
    """
    flattened_X = tf.reshape(X, [-1, W.shape[0]]) # (256, 1024*3)
    return softmax(tf.matmul(flattened_X ,W) +b)

def sgd(params, grads, lr, bs):
    """
    stochastic gradient descent implementation
    args:
    - params [list[tensor]]: model params
    - grad [list[tensor]]: param gradient such that params[0].shape == grad[0].shape
    - lr [float]: learning rate
    - bs [int]: batch_size
    """
    # IMPLEMENT THIS FUNCTION
    for param, grad in zip(params,grads):
        param.assign_sub(lr * grad / bs)



def training_loop(train_dataset, W, b, model, lossFn, optimizerFn):
    """
    training loop
    args:
    - train_dataset: 
    - model [func]: model function
    - loss [func]: loss function
    - optimizer [func]: optimizer func
    returns:
    - mean_loss [tensor]: mean training loss
    - mean_acc [tensor]: mean training accuracy
    """
    accuracies = []
    losses = []
    lr = 0.1
    
    # loop over all batches
    for X, Y in train_dataset:
        with tf.GradientTape() as tape:
            # IMPLEMENT THIS FUNCTION
            with tf.GradientTape() as tape:
                X = X/255.0 # why ??
                y_ohe = tf.one_hot(Y, 43)
                y_hat = model(W, X, b)
                loss = lossFn(y_hat, y_ohe) # shape = (batch_size, )
                losses.append(tf.math.reduce_mean(loss))
            
                params = [W,b]
                gradients = tape.gradient(loss, [W,b])
                optimizerFn([W,b], gradients, lr, X.shape[0])

                acc = accuracy(y_hat, Y)
                accuracies.append(acc)
            
            
    mean_acc = tf.math.reduce_mean(tf.concat(accuracies, axis=0))
    mean_loss = tf.math.reduce_mean(losses)
    return mean_loss, mean_acc


def validation_loop(val_dataset, W, b, model):
    """
    training loop
    args:
    - train_dataset: 
    - W
    - b
    - model [func]: model function
    - loss [func]: loss function
    - optimizer [func]: optimizer func
    returns:
    - mean_acc [tensor]: mean validation accuracy
    """
    # IMPLEMENT THIS FUNCTION
    accuracies = []
    for X, Y in val_dataset:
        X = X / 255.0
        y_hat  = model(W, X, b)
        acc = accuracy(y_hat, Y)
        accuracies.append(acc)
    mean_acc = tf.math.reduce_mean(accuracies)
    mean_acc2 = tf.math.reduce_mean(tf.concat(accuracies, axis=0))
    return mean_acc


if __name__  == '__main__':
    logger = get_module_logger(__name__)
    parser = argparse.ArgumentParser(description='Download and process tf files')
    parser.add_argument('--imdir', required=True, type=str,
                        help='data directory')
    parser.add_argument('--epochs', default=10, type=int,
                        help='Number of epochs')
    args = parser.parse_args()    

    logger.info(f'Training for {args.epochs} epochs using {args.imdir} data')
    # get the datasets
    train_dataset, val_dataset = get_datasets(args.imdir)

    # set the variables
    num_inputs = 1024*3 # image_shape(32,32,3) => flatten to 1024*3
    num_outputs = 43
    W = tf.Variable(tf.random.normal(shape=(num_inputs, num_outputs),
                                    mean=0, stddev=0.01))
    b = tf.Variable(tf.zeros(num_outputs))

    # training! 
    for epoch in range(args.epochs):
        logger.info(f'Epoch {epoch}')
        loss, acc = training_loop(train_dataset, W, b, model, cross_entropy, sgd)
        logger.info(f'Mean training loss: {loss:1f}, mean training accuracy {acc:1f}')
        acc = validation_loop(val_dataset, W, b, model)
        logger.info(f'Mean validation accuracy {acc:1f}')

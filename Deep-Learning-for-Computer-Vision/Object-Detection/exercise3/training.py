import argparse
import logging

import tensorflow as tf
from tensorflow.keras.optimizers import schedules
from utils import get_datasets, get_module_logger, display_metrics, \
    create_network, LrLogger


def exponential_decay(model, callbacks, lr=0.001):
    """ use exponential decay """
    # IMPLEMENT THIS FUNCTION
    
    scheduler = tf.keras.optimizers.schedules.ExponentialDecay(lr, decay_steps=100, decay_rate=0.94)
    #APPROACH 1: pass callback to optimizer directlyy. No callback update needed
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=scheduler),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model, callbacks


def step_decay(model, callbacks, lr=0.001):
    """ create custom decay using learning rate scheduler """
    #  IMPLEMENT THIS FUNCTION
    #  LearningRateScheduler
    def scheduler(epoch, lr):
        if epoch % 5 == 0 and epoch > 0:
            
            lr /= 2
        return lr

    # APPROACH 2 : append to callbacks []
    callbacks.append(tf.keras.callbacks.LearningRateScheduler(scheduler))
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    return model, callbacks


if __name__  == '__main__':
    logger = get_module_logger(__name__)
    parser = argparse.ArgumentParser(description='Download and process tf files')
    parser.add_argument('-d', '--imdir', required=True, type=str,
                        help='data directory')
    parser.add_argument('-e', '--epochs', default=10, type=int,
                        help='Number of epochs')
    args = parser.parse_args()    

    logger.info(f'Training for {args.epochs} epochs using {args.imdir} data')
    # get the datasets
    train_dataset, val_dataset = get_datasets(args.imdir)
    logger = LrLogger() # Custom class - to Log learning rates after every epoch
    callbacks = [logger]

    model = create_network()

    model, callbacks = exponential_decay(model, callbacks)
#     model, callbacks = step_decay(model, callbacks)

    history = model.fit(x=train_dataset, 
                        epochs=args.epochs, 
                        validation_data=val_dataset,
                        callbacks=callbacks)
    display_metrics(history)
import numpy as np
import tensorflow as tf
from scipy.stats import kendalltau
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import wandb
import math


class Constants:
    DATA_DIR = 'data'


def kendall_tau_per_query(y_pred, y, q, ds="train"):
    tau_list = []
    for qi in np.random.choice(np.unique(q), size=1000, replace=False):
        y_q = y[q == qi]
        y_pred_q = tf.nn.softmax(y_pred[q == qi], axis=0)
        tau_list.append(kendalltau(y_q, y_pred_q)[0])

        if math.isnan(tau_list[-1]):
            print("Found NaN\n--y_pred--\n{}\n--y--\n{}\n".format(y_pred_q, y_q))

    tau_mean = np.mean(tau_list)
    tau_std = np.std(tau_list)

    wandb.log({'tau_mean_{}'.format(ds): tau_mean})
    wandb.log({'tau_std_{}'.format(ds): tau_std})

    return tau_mean, tau_std


class PrintKendalTau(keras.callbacks.Callback):

    def __init__(self, eval_data):
        self.generator = eval_data

    def log_train_tau(self, epoch):
        y_pred = self.model.predict(self.generator.train_data[0])
        y = self.generator.train_data[1]
        q = self.generator.train_data[2]

        tau = kendall_tau_per_query(y_pred, y, q, ds="train")

        print("\nEpoch: {} - Train Data - Kendal Tau: {}".format(epoch, tau))

    def log_test_tau(self, epoch):
        y_pred = self.model.predict(self.generator.test_data[0])
        y = self.generator.test_data[1]
        q = self.generator.test_data[2]

        tau = kendall_tau_per_query(y_pred, y, q, ds="test")

        print("\nEpoch: {} - Test Data - Kendal Tau: {}".format(epoch, tau))


    def on_epoch_end(self, epoch, logs=None):
        self.log_train_tau(epoch)
        self.log_test_tau(epoch)


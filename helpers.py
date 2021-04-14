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
    num_queries = len(np.unique(q)) if len(np.unique(q)) < 1000 else 1000
    for qi in np.random.choice(np.unique(q), size=num_queries, replace=False):
        y_q = y[q == qi]
        y_pred_q = tf.nn.softmax(y_pred[q == qi], axis=0)

        if len(np.unique(y_pred_q)) == 1:
            tau_list.append(0.0)
        else:
            tau_list.append(kendalltau(y_q, y_pred_q)[0])

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

        print("Epoch: {} - Train Data - Kendal Tau: {}".format(epoch, tau))

    def log_test_tau(self, epoch):
        y_pred = self.model.predict(self.generator.test_data[0])
        y = self.generator.test_data[1]
        q = self.generator.test_data[2]

        tau = kendall_tau_per_query(y_pred, y, q, ds="test")

        print("Epoch: {} - Test Data - Kendal Tau: {}".format(epoch, tau))


    def on_epoch_end(self, epoch, logs=None):
        print()
        self.log_train_tau(epoch)
        self.log_test_tau(epoch)

class PrintTensor(keras.callbacks.Callback):

    def __init__(self, eval_data, tensor):
        self.generator = eval_data
        self.output_tensor = tensor

    def on_epoch_begin(self, epoch, logs=None):
        print()
        #x, y, q = self.generator.test_data[0], self.generator.test_data[1], self.generator.test_data[2]
        data, yi = self.generator.make_batch_listnet(1)
        #data = x[q==42]
        #yi = y[q==42]
        y_pred = self.output_tensor(data).numpy().astype(np.double)
        y_actual = tf.nn.softmax(yi.astype(np.double))
        y_pred_scores = tf.nn.softmax(y_pred, axis=0)
        y_actual_scores = tf.nn.softmax(y_actual, axis=0)
        print('Output tensor, no activation: {}'.format(self.output_tensor(data)))
        print('Output tensor, softmax: {}'.format(tf.nn.softmax(self.output_tensor(data), axis=0)))
        print('Labels, no activation: {}'.format(yi))
        print('Labels, softmax: {}'.format(tf.nn.softmax(yi.astype(float), axis=0)))
        #loss = -tf.reduce_sum(y_actual_scores * tf.math.log(y_pred_scores))
        loss = self._def_cost(y_actual, y_pred)
        print('Loss value: {}'.format(loss))

    #def on_batch_begin(self, batch, logs=None):
    #    print(batch)
    #    exit(1)

    def _def_cost(self, y_actual, y_pred):
        """
        The Top-1 approximated ListNet loss as in Cao et al (2006), Learning to
        Rank: From Pairwise Approach to Listwise Approach
        :param nn: activation of the previous layer
        :param y: target labels
        :return: The loss
        """
        # ListNet top-1 reduces to a softmax and simple cross entropy
        y_actual_scores = tf.nn.softmax(y_actual, axis=0)
        y_pred_scores = tf.nn.softmax(y_pred, axis=0)
        #K.print_tensor(y_actual_scores, message='y_actual')
        #K.print_tensor(y_pred_scores, message='y_pred')
        return -tf.reduce_sum(y_actual_scores * tf.math.log(y_pred_scores))
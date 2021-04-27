import numpy as np
import tensorflow as tf
from scipy.stats import kendalltau
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import wandb
import math
import os


class Constants:
    DATA_DIR = 'data'


def kendall_tau_per_query(y_pred, y, q):
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

    return tau_mean, tau_std


class PrintKendalTau(keras.callbacks.Callback):

    def __init__(self, eval_data):
        self.generator = eval_data

    def log_tau(self, epoch, ds, ds_name):
        y_pred = self.model.predict(ds[0])
        y = ds[1]
        q = ds[2]

        tau_mean, tau_std = kendall_tau_per_query(y_pred, y, q)

        wandb.log({'epoch': epoch,
                   'tau_mean_{}'.format(ds_name): tau_mean,
                   'tau_std_{}'.format(ds_name): tau_std })

    def on_epoch_end(self, epoch, logs=None):
        self.log_tau(epoch, self.generator.train_data, "train")
        self.log_tau(epoch, self.generator.dev_data, "validation")

class PrintTensor(keras.callbacks.Callback):

    def __init__(self, eval_data, tensor):
        self.generator = eval_data
        self.output_tensor = tensor

    def log_predictions_to_wandb(self, X,y,q):
        table = wandb.Table(columns=["epoch", "ex_id", "y_actual", "y_pred"])

        for i in set(q):
                qi = q == i

                data = x[qi]
                yi = y[qi]

                y_pred = self.output_tensor(data).numpy().astype(np.double)
                y_actual = tf.nn.softmax(yi.astype(np.double))
                y_pred_scores = tf.argsort(tf.reshape(
                    tf.nn.softmax(y_pred, axis=0), [-1]))
                ex_len = y_pred_scores.shape[0]

                y_pred_scores = tf.reshape(y_pred_scores, [ex_len, 1])
                y_actual_scores = tf.argsort(tf.nn.softmax(y_actual, axis=0))
                y_actual_scores = tf.reshape(y_actual_scores, [ex_len, 1])

                exno = i

                ep_data = np.zeros([ex_len, 1]) + epoch
                ex_data = np.zeros([ex_len, 1]) + exno

                data = np.concatenate(
                    [ep_data, ex_data, y_actual_scores, y_pred_scores], axis=1)

                for d in data:
                    table.add_data(*d)

        wandb.log({"examples": table}, commit = False)

    def on_epoch_begin(self, epoch, logs=None):
        if epoch % 10 != 1:
            return

        # x, y, q = self.generator.dev_data
        # self.log_predictions_to_wandb(X,y,q)

        # These will be uploaded at the end of training
        self.model.save(os.path.join(wandb.run.dir,"model_ckpt_{}.h5".format(epoch-1)))
        



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

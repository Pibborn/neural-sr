import numpy as np
import tensorflow as tf
from scipy.stats import kendalltau
import matplotlib.pyplot as plt
import wandb

hist_count = 0

def kendall_tau_per_query(y_pred, y, q):
    tau_list = []
    for qi in np.unique(q):
        y_q = y[q == qi]
        y_pred_q = tf.nn.softmax(y_pred[q == qi], axis=0)
        tau_list.append(kendalltau(y_q, y_pred_q)[0])

    data = [[s] for s in tau_list]
    table = wandb.Table(data=data, columns=["taus"])
    wandb.log({'tau_distribution_{}'.format(hist_count): wandb.plot.histogram(table, "taus", title=None)})
    hist_count+=1

    return np.mean(tau_list), np.std(tau_list)


class Constants:
    DATA_DIR = 'data'

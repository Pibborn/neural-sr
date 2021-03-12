import numpy as np
import tensorflow as tf
from scipy.stats import kendalltau
import matplotlib.pyplot as plt

def kendall_tau_per_query(y_pred, y, q):
    tau_list = []
    for qi in np.unique(q):
        y_q = y[q == qi]
        y_pred_q = tf.nn.softmax(y_pred[q == qi], axis=0)
        tau_list.append(kendalltau(y_q, y_pred_q))
    plt.hist(tau_list, bins=np.arange(-1., 1., 0.1), histtype='step')
    plt.savefig('hist.png')
    return np.mean(tau_list), np.std(tau_list)


class Constants:
    DATA_DIR = 'data'

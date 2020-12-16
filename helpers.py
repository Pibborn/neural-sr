import numpy as np
from scipy.stats import kendalltau

def kendall_tau_per_query(y, y_pred, q):
    tau_list = []
    for qi in np.unique(q):
        y_q = y[q == qi]
        y_pred_q = y_pred[q == qi]
        tau_list.append(kendalltau(y_q, y_pred_q))
    return np.mean(tau_list), np.std(tau_list)


class Constants:
    DATA_DIR = 'data'

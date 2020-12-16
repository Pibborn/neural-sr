colab = False # SET THIS
wandb = False # SET THIS

from data.srst19.loader import load_data
from models.DirectRanker import DirectRanker
from helpers import kendall_tau_per_query
import wandb

if wandb:
    wandb.init(project="neural-sr", entity="Pibborn", sync_tensorboard=True)

if __name__ == '__main__':
    (x_train, y_train, q_train), (x_dev, y_dev, q_dev), (x_test, y_test, q_test) = load_data('en')
    dr = DirectRanker(num_features=len(x_train[0]), batch_size=64, hidden_layers_dr=[30, 20], query=True)
    dr.fit(x_train, y_train, epochs=30, q=q_train)
    y_pred = dr.predict_proba(x_test)
    avg_tau, std_tau = kendall_tau_per_query(y_pred, y_test, q_test)
    print('Average Kendall tau: {} \n Standard Deviation: {}'.format(avg_tau, std_tau))
    wandb.log({'KTau avg': avg_tau, 'KTau std': std_tau})

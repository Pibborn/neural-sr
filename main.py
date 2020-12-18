colab = False # SET THIS
wandb = False # SET THIS

from data.srst19.loader import load_data, SRST19Generator
from models.DirectRanker import DirectRanker
from helpers import kendall_tau_per_query
import wandb

if wandb:
    wandb.init(project="neural-sr", entity="Pibborn", sync_tensorboard=True)

if __name__ == '__main__':
    train_gen = SRST19Generator(language='en', split='train')
    #val_gen = SRST19Generator(language='en', split='dev')
    num_features = len(train_gen.train_data[0][0])
    dr = DirectRanker(num_features=num_features, batch_size=1024, query=True, epoch=50, verbose=1)
    dr.fit(train_gen)
    y_pred = dr.predict_proba(train_gen.test_data[0])
    avg_tau, std_tau = kendall_tau_per_query(y_pred, train_gen.test_data[1], train_gen.test_data[2])
    print('Average Kendall tau: {} \n Standard Deviation: {}'.format(avg_tau, std_tau))
    wandb.log({'KTau avg': avg_tau, 'KTau std': std_tau})

colab = False # SET THIS
wandb = False # SET THIS

from loader import DatasetGenerator
from models.DirectRanker import DirectRanker
from models.ListNet import ListNet
from helpers import kendall_tau_per_query
import wandb

if wandb:
    wandb.init(project="neural-sr", entity="jgu-wandb", sync_tensorboard=True)

if __name__ == '__main__':
    train_gen = DatasetGenerator("srst19", language='en', split='train', pairwise=False)
    #val_gen = DatasetGenerator(language='en', split='dev')
    print('Generator loaded.')
    num_features = len(train_gen.train_data[0][0])
    dr = ListNet(num_features=num_features, batch_size=1024, epoch=300, verbose=1, learning_rate_decay_rate=0)
    dr.fit(train_gen)
    y_pred = dr.predict_proba(train_gen.test_data[0])
    avg_tau, std_tau = kendall_tau_per_query(y_pred, train_gen.test_data[1], train_gen.test_data[2])
    print('Average Kendall tau: {} \n Standard Deviation: {}'.format(avg_tau, std_tau))
    wandb.log({'KTau avg': avg_tau, 'KTau std': std_tau})

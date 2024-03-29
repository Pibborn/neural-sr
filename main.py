colab = False # SET THIS
wandb = True # SET THIS

from loader import DatasetGenerator
from models.DirectRanker import DirectRanker
from models.ListNet import ListNet
from models.PointwiseNet import PointwiseNet
from helpers import kendall_tau_per_query
import wandb
import os

# Disabling GPU computation since is not useful with these experiments
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if wandb:
    wandb.init(
        project="neural-sr", 
        entity="jgu-wandb", 
        sync_tensorboard=True,
        allow_val_change=True)

if __name__ == '__main__':
    train_gen = DatasetGenerator(wandb.config.dataset, split='train', pairwise=wandb.config.pairwise,
                                 limit_dataset_size=wandb.config.limit_dataset_size, query=wandb.config.query)
                                 #val_gen = DatasetGenerator(language='en', split='dev')


    num_features = len(train_gen.train_data[0][0])
    if not wandb.config.query:
        print('Using pointwise ranker.')
        dr = PointwiseNet(
            num_features=num_features,
            batch_size=wandb.config.batch_size,
            epoch=wandb.config.epoch,
            verbose=1,
            learning_rate_decay_rate=0,
            feature_activation_dr=wandb.config.feature_activation,
            kernel_regularizer_dr=wandb.config.regularization,
            learning_rate=wandb.config.learning_rate,
            hidden_layers_dr=wandb.config.hidden_layers
        )
    else:
        if wandb.config.pairwise:
            print('Using DirectRanker.')
            dr = DirectRanker(
                num_features=num_features,
                batch_size=wandb.config.batch_size,
                epoch=wandb.config.epoch,
                verbose=1,
                learning_rate_decay_rate=0,
                feature_activation_dr=wandb.config.feature_activation,
                kernel_regularizer_dr=wandb.config.regularization,
                learning_rate=wandb.config.learning_rate,
                hidden_layers_dr=wandb.config.hidden_layers)
        else:
            print('Using ListNet.')
            dr = ListNet(
                        num_features=num_features,
                        batch_size=wandb.config.batch_size,
                        epoch=wandb.config.epoch,
                        verbose=1,
                        learning_rate_decay_rate=0,
                        feature_activation_dr=wandb.config.feature_activation,
                        kernel_regularizer_dr=wandb.config.regularization,
                        learning_rate=wandb.config.learning_rate,
                        hidden_layers_dr=wandb.config.hidden_layers)
    dr.fit(train_gen)

    # y_pred = dr.predict_proba(train_gen.test_data[0])
    dr.model.save(os.path.join(wandb.run.dir, "model.h5"))

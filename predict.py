from loader import DatasetGenerator
from models.DirectRanker import DirectRanker
from models.ListNet import ListNet
from helpers import kendall_tau_per_query
import wandb
import os

if wandb:
    wandb.init(
        project="neural-sr", 
        entity="jgu-wandb", 
        config="config-current-run.yml",
        sync_tensorboard=True,
        allow_val_change=True)

if __name__ == '__main__':
    train_gen = DatasetGenerator(wandb.config["dataset"], language='en', split='train', pairwise=False, limit_dataset_size=wandb.config.limit_dataset_size)
                                 #val_gen = DatasetGenerator(language='en', split='dev')

    num_features = len(train_gen.train_data[0][0])
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

    wandb.restore('model.h5', run_path="jgu-wandb/neural-sr/3kldppmo")
    dr.model.load_weights('model.h5')

    y_pred = dr.predict_proba(train_gen.test_data[0])


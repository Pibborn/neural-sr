colab = False # SET THIS
wandb = False # SET THIS

from wandb.env import ARGS
from loader import DatasetGenerator
from models.DirectRanker import DirectRanker
from models.ListNet import ListNet
from helpers import kendall_tau_per_query
import wandb
import os
import yaml
import sys

# Disabling GPU computation since is not useful with these experiments
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

restored_config = wandb.restore(
    'config.yaml', run_path=sys.argv[1], replace=True, root=os.path.join("tmp", sys.argv[1]))

# Working around bug in wandb which does not allow to change config values
# even when allow_val_change is set to True. To workaround that, we remove
# the config-defaults.yaml file from the directory, download the config file
# from the network, update it to remove the limit_dataset_size constraint and
# use the modified file to initialize wandb.
with open(restored_config.name) as file:
    config = yaml.safe_load(file)

config['limit_dataset_size'] = { 'desc': None, 'value': None }
config['target-run'] = { 'desc': 'wandb run used to initialize training', 'value': sys.argv[1] }
config['epoch']['value'] = 200

WORKAROUND_CONFIG_FILE = "full-train-config.yaml"
with open(WORKAROUND_CONFIG_FILE, 'w') as file:
    yaml.dump(config, stream=file)

if wandb:
    wandb.init(
        project="neural-sr", 
        entity="jgu-wandb",
        config=WORKAROUND_CONFIG_FILE,
        sync_tensorboard=True,
        allow_val_change=True)

if __name__ == '__main__':
    train_gen = DatasetGenerator(wandb.config.dataset, split='train', pairwise=False, limit_dataset_size=wandb.config.limit_dataset_size)
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
    dr.fit(train_gen)

    # y_pred = dr.predict_proba(train_gen.test_data[0])
    dr.model.save(os.path.join(wandb.run.dir, "model.h5"))

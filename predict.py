from loader import DatasetGenerator
from models.DirectRanker import DirectRanker
from models.ListNet import ListNet
from helpers import kendall_tau_per_query
import wandb
import os
import yaml
import pandas as pd
import numpy as np
import tensorflow as tf
import sys


def predictions_to_pandas(model, x, y, q):
    print("predicting")
    if model.name == 'ListNet':
        y_ = model.predict_proba(x).numpy().astype(np.double)
    else:
        y_ = model.predict_proba(x).astype(np.double)
    table = pd.DataFrame(columns=["ex_id", "y_actual", "y_pred"])

    print(print(kendall_tau_per_query(y_, y, q)))

    print("formatting output")
    for i in set(q):
            qi = q == i

            data = x[qi]
            yi = y[qi]
            y_i = y_[qi]

            ex_len = data.shape[0]

            y_pred_scores = tf.argsort(tf.reshape(tf.nn.softmax(y_i, axis=0), [-1]))
            y_pred_scores = tf.reshape(y_pred_scores, [ex_len, 1])

            y_actual = tf.nn.softmax(yi.astype(np.double))
            y_actual_scores = tf.argsort(tf.nn.softmax(y_actual, axis=0))
            y_actual_scores = tf.reshape(y_actual_scores, [ex_len, 1])

            exno = i

            ex_data = np.zeros([ex_len, 1]) + exno

            all_out_data = np.concatenate(
                [ex_data, y_actual_scores, y_pred_scores], axis=1)

            panda_out = pd.DataFrame(all_out_data.astype(int), columns=table.columns)
            table = table.append(panda_out)

    return table

# Disabling GPU computation since is not useful with these experiments
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

restored_model = wandb.restore(
    'model.h5', run_path=sys.argv[1], replace=True, root=os.path.join("tmp", sys.argv[1]))
restored_config = wandb.restore(
    'config.yaml', run_path=sys.argv[1], replace=True, root=os.path.join("tmp", sys.argv[1]))

with open(restored_config.name) as file:
    config = yaml.safe_load(file)

config['limit_dataset_size'] = { 'desc': None, 'value': None }
config['target-run'] = { 'desc': 'wandb run used to restore the model', 'value': sys.argv[1] }
config['pairwise'] = {'desc': 'use pairwise model', 'value': sys.argv[2]}

WORKAROUND_CONFIG_FILE = "predict-config.yaml"
with open(WORKAROUND_CONFIG_FILE, 'w') as file:
    yaml.dump(config, stream=file)

run = wandb.init(
        project="neural-sr", 
        entity="jgu-wandb",
        config=WORKAROUND_CONFIG_FILE,
        job_type="model-test")

if __name__ == '__main__':
    # note that at test time the DirectRanker does not need paired data points, hence pairwise=False
    train_gen = DatasetGenerator(wandb.config.dataset, split='train', pairwise=False, limit_dataset_size=wandb.config.limit_dataset_size)
                                 #val_gen = DatasetGenerator(language='en', split='dev')

    num_features = len(train_gen.train_data[0][0])
    if not wandb.config['pairwise']:
        dr = ListNet(
                    num_features=num_features, 
                    batch_size=512, 
                    epoch=wandb.config.epoch,
                    verbose=1, 
                    learning_rate_decay_rate=0, 
                    feature_activation_dr=wandb.config.feature_activation, 
                    kernel_regularizer_dr=wandb.config.regularization,
                    learning_rate=wandb.config.learning_rate,
                    hidden_layers_dr=wandb.config.hidden_layers)
    else:
        print('Using DirectRanker.')
        dr = DirectRanker(
                    num_features=num_features, 
                    batch_size=512, 
                    epoch=wandb.config.epoch,
                    verbose=1, 
                    learning_rate_decay_rate=0, 
                    feature_activation_dr=wandb.config.feature_activation, 
                    kernel_regularizer_dr=wandb.config.regularization,
                    learning_rate=wandb.config.learning_rate,
                    hidden_layers_dr=wandb.config.hidden_layers)

    dr._build_model()
    dr.model.summary()

    dr.model.load_weights(restored_model.name)
    dr.verbose = False

    dataset_name = wandb.config.dataset.split('/')[1]
    artifact = wandb.Artifact('{}_predictions'.format(dataset_name), type="predictions")


    for item in [(train_gen.dev_data, "validation"), (train_gen.test_data, "test"), (train_gen.train_data, "train")]:
        ds, label = item
        X,y,q = ds

        dataframe = predictions_to_pandas(dr, X,y,q)
        predictions_file = "predictions_{}_pairwise_{}.csv".format(label, wandb.config['pairwise'])
        dataframe.to_csv(predictions_file)

        artifact.add_file(predictions_file)

    run.log_artifact(artifact)


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


def predictions_to_pandas(model, x, y, q):
    table = pd.DataFrame(columns=["ex_id", "y_actual", "y_pred"])

    for i in set(q):
            qi = q == i

            data = x[qi]
            yi = y[qi]

            y_pred = model.predict_proba(data).numpy().astype(np.double)
            y_actual = tf.nn.softmax(yi.astype(np.double))
            y_pred_scores = tf.argsort(tf.reshape(
                tf.nn.softmax(y_pred, axis=0), [-1]))
            ex_len = y_pred_scores.shape[0]

            y_pred_scores = tf.reshape(y_pred_scores, [ex_len, 1])
            y_actual_scores = tf.argsort(tf.nn.softmax(y_actual, axis=0))
            y_actual_scores = tf.reshape(y_actual_scores, [ex_len, 1])

            exno = i

            ex_data = np.zeros([ex_len, 1]) + exno

            all_out_data = np.concatenate(
                [ex_data, y_actual_scores, y_pred_scores], axis=1)

            panda_out = pd.DataFrame(all_out_data, columns=table.columns)
            table.append(panda_out)

            # for d in data:
            #     table.append(*d)

    return table

restored_model = wandb.restore(
    'model.h5', run_path="jgu-wandb/neural-sr/3kldppmo")
restored_config = wandb.restore(
    'config.yaml', run_path="jgu-wandb/neural-sr/3kldppmo")

with open(restored_config.name) as file:
    config = yaml.safe_load(file)

if __name__ == '__main__':
    train_gen = DatasetGenerator(config["dataset"]["value"], language='en', split='train', pairwise=False, limit_dataset_size=config['limit_dataset_size']["value"])
                                 #val_gen = DatasetGenerator(language='en', split='dev')

    num_features = len(train_gen.train_data[0][0])
    dr = ListNet(
                num_features=num_features, 
                batch_size=config['batch_size']["value"], 
                epoch=config['epoch']["value"],
                verbose=1, 
                learning_rate_decay_rate=0, 
                feature_activation_dr=config['feature_activation']["value"], 
                kernel_regularizer_dr=config['regularization']["value"],
                learning_rate=config['learning_rate']["value"],
                hidden_layers_dr=config['hidden_layers']["value"])


    dr._build_model()
    dr.model.summary()

    dr.model.load_weights(restored_model.name)
    dr.verbose = False

    X,y,q = train_gen.test_data

    df = predictions_to_pandas(dr, X,y,q)
    df.to_csv("predictions.csv")


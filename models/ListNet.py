
import tensorflow as tf

import numpy as np
from sklearn.base import BaseEstimator
from helpers import kendall_tau_per_query
import tensorflow.keras as keras


class PrintKendalTau(keras.callbacks.Callback):

    def __init__(self, eval_data):
        self.generator = eval_data

    def kendal_metric(q):
        def kendal_tau(y_true, y_pred):
            return kendall_tau_per_query(y_pred.numpy(), y_true.numpy(), q)

        if q is None:
            return 'acc'

        return kendal_tau

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.generator.train_data[0])
        y_pred = tf.nn.sigmoid(y_pred) # implements predict_proba
        tau = kendall_tau_per_query(y_pred, self.generator.train_data[1], self.generator.train_data[2])
        print("\nEpoch: {} Kendal Tau: {}".format(epoch, tau))

class ListNet(BaseEstimator):
    """
    Tensorflow implementation of https://arxiv.org/pdf/1805.08716.pdf
    Inspired by: https://github.com/MilkaLichtblau/DELTR-Experiments

    Constructor
    :param hidden_layers: List containing the numbers of neurons in the layers for feature
    :param activation: tf function for the feature part of the net
    :param kernel_initializer: tf kernel_initializer
    :param start_batch_size: cost function of FairListNet
    :param min_doc: min size of docs in query if a list is given
    :param end_batch_size: max size of docs in query if a list is given
    :param start_len_qid: start size of the queries/batch
    :param end_len_qid: end size of the queries/batch
    :param learning_rate: learning rate for the optimizer
    :param max_steps: total training steps
    :param learning_rate_step_size: factor for increasing the learning rate
    :param learning_rate_decay_factor: factor for increasing the learning rate
    :param optimizer: tf optimizer object
    :param print_step: for which step the script should print out the cost for the current batch
    :param weight_regularization: float for weight regularization
    :param dropout: float amount of dropout
    :param input_dropout: float amount of input dropout
    :param name: name of the object
    :param num_features: number of input features
    :param protected_feature_deltr: column name of the protected attribute (index after query and document id)
    :param gamma_deltr: value of the gamma parameter
    :param iterations_deltr: number of iterations the training should run
    :param standardize_deltr: let's apply standardization to the features
    :param random_seed: random seed
    """

    def __init__(self,
                 # ListNet HPs
                 hidden_layers_dr=[256, 128, 64, 20],
                 feature_activation_dr='tanh',
                 ranking_activation_dr='sigmoid',
                 feature_bias_dr=True,
                 kernel_initializer_dr=tf.random_normal_initializer,
                 kernel_regularizer_dr=0.0,
                 # Common HPs
                 batch_size=200,
                 learning_rate=0.001,
                 learning_rate_decay_rate=1,
                 learning_rate_decay_steps=1000,
                 optimizer=tf.keras.optimizers.Adam,# 'Nadam' 'SGD'
                 epoch=10,
                 # other variables
                 verbose=0,
                 validation_size=0.0,
                 num_features=0,
                 random_seed=42,
                 name="ListNet",
                 dtype=tf.float32,
                 print_summary=False,
                 ):

        # ListNet HPs
        self.hidden_layers_dr = hidden_layers_dr
        self.feature_activation_dr = feature_activation_dr
        self.ranking_activation_dr = ranking_activation_dr
        self.feature_bias_dr = feature_bias_dr
        self.kernel_initializer_dr = kernel_initializer_dr
        self.kernel_regularizer_dr = kernel_regularizer_dr
        # Common HPs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_decay_rate = learning_rate_decay_rate
        self.learning_rate_decay_steps = learning_rate_decay_steps
        self.optimizer = optimizer
        self.epoch = epoch
        # other variables
        self.verbose = verbose
        self.validation_size = validation_size
        self.num_features = num_features
        self.random_seed = random_seed
        self.name = name
        self.dtype = dtype
        self.print_summary = print_summary

    def _build_model(self):
        """
        This function builds the ListNet with the values specified in the constructor
        :return:
        """

        # Placeholders for the inputs
        input_layer = tf.keras.layers.Input(
            shape=self.num_features,
            dtype=self.dtype,
            name="input"
        )

        nn = tf.keras.layers.Dense(
            units=self.hidden_layers_dr[0],
            activation=self.feature_activation_dr,
            use_bias=self.feature_bias_dr,
            kernel_initializer=self.kernel_initializer_dr(seed=self.random_seed),
            kernel_regularizer=tf.keras.regularizers.l2(self.kernel_regularizer_dr),
            bias_regularizer=tf.keras.regularizers.l2(self.kernel_regularizer_dr),
            activity_regularizer=tf.keras.regularizers.l2(self.kernel_regularizer_dr),
            name="nn_hidden_0"
        )(input_layer)

        for i in range(1, len(self.hidden_layers_dr)):
            nn = tf.keras.layers.Dense(
                units=self.hidden_layers_dr[i],
                activation=self.feature_activation_dr,
                use_bias=self.feature_bias_dr,
                kernel_initializer=self.kernel_initializer_dr(seed=self.random_seed),
                kernel_regularizer=tf.keras.regularizers.l2(self.kernel_regularizer_dr),
                bias_regularizer=tf.keras.regularizers.l2(self.kernel_regularizer_dr),
                activity_regularizer=tf.keras.regularizers.l2(self.kernel_regularizer_dr),
                name="nn_hidden_" + str(i)
            )(nn)

        nn = tf.keras.layers.Dense(
            units=1,
            activation=None,
            kernel_initializer=self.kernel_initializer_dr(seed=self.random_seed),
            kernel_regularizer=tf.keras.regularizers.l2(self.kernel_regularizer_dr),
            bias_regularizer=tf.keras.regularizers.l2(self.kernel_regularizer_dr),
            activity_regularizer=tf.keras.regularizers.l2(self.kernel_regularizer_dr),
            name="nn_cls"
        )(nn)

        self.model = tf.keras.models.Model(input_layer, nn, name='ListNet')

        if self.learning_rate_decay_steps > 0:
            lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
                self.learning_rate,
                decay_steps=self.learning_rate_decay_steps,
                decay_rate=self.learning_rate_decay_rate,
                staircase=False
            )
        else:
            lr_schedule = self.learning_rate

        self.model.compile(
            optimizer=self.optimizer(lr_schedule),
            loss=self._def_cost,
            metrics=[]
        )

        if self.print_summary:
            self.model.summary()

    def _def_cost(self, y_actual, y_pred):
        """
        The Top-1 approximated ListNet loss as in Cao et al (2006), Learning to
        Rank: From Pairwise Approach to Listwise Approach
        :param nn: activation of the previous layer
        :param y: target labels
        :return: The loss
        """
        # ListNet top-1 reduces to a softmax and simple cross entropy
        y_actual_scores = tf.nn.softmax(y_actual, axis=0)
        y_pred_scores = tf.nn.softmax(y_pred, axis=0)
        return -tf.reduce_sum(y_actual_scores * tf.math.log(y_pred_scores))

    def fit(self, generator, **fit_params):
        """
        :param features: list of queries for training the net
        :param real_classes: list of labels inside a query
        :return:
        """
        self._build_model()

        self.model.fit_generator(
            generator=generator,
            epochs=self.epoch,
            verbose=self.verbose,
            workers=1,
            callbacks=[PrintKendalTau(generator)]
        )

    def predict_proba(self, features):
        """
        :param features: list of features of the instance feed to the net
        :return: predicted class
        """
        if len(features.shape) == 1:
            features = [features]

        res = self.model.predict(features, batch_size=self.batch_size, verbose=self.verbose)

        return tf.nn.sigmoid(res)

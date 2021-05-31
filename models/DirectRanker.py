import numpy as np
import tensorflow as tf
from helpers import PrintKendalTau, PrintTensor
from sklearn.base import BaseEstimator

class DirectRanker(BaseEstimator):
    """
    TODO
    """

    def __init__(self,
                 # DirectRanker HPs
                 hidden_layers_dr=[256, 128, 64, 20],
                 feature_activation_dr='tanh',
                 ranking_activation_dr='sigmoid',
                 feature_bias_dr=True,
                 kernel_initializer_dr=tf.random_normal_initializer,
                 kernel_regularizer_dr=0.0,
                 drop_out=0,
                 # Common HPs
                 scale_factor_train_sample=5,
                 batch_size=200,
                 loss=tf.keras.losses.MeanSquaredError(),# 'binary_crossentropy'
                 learning_rate=0.001,
                 learning_rate_decay_rate=1,
                 learning_rate_decay_steps=1000,
                 optimizer=tf.keras.optimizers.Adam,# 'Nadam' 'SGD'
                 epoch=10,
                 steps_per_epoch=None,
                 # other variables
                 verbose=0,
                 validation_size=0.0,
                 num_features=0,
                 random_seed=42,
                 name="DirectRanker",
                 dtype=tf.float32,
                 print_summary=False,
                 query=False
                 ):

        # DirectRanker HPs
        self.hidden_layers_dr = hidden_layers_dr
        self.feature_activation_dr = feature_activation_dr
        self.ranking_activation_dr = ranking_activation_dr
        self.feature_bias_dr = feature_bias_dr
        self.kernel_initializer_dr = kernel_initializer_dr
        self.kernel_regularizer_dr = kernel_regularizer_dr
        self.drop_out = drop_out
        # Common HPs
        self.scale_factor_train_sample = scale_factor_train_sample
        self.batch_size = batch_size
        self.loss = loss
        self.learning_rate = learning_rate
        self.learning_rate_decay_rate = learning_rate_decay_rate
        self.learning_rate_decay_steps = learning_rate_decay_steps
        self.optimizer = optimizer
        self.epoch = epoch
        self.steps_per_epoch = steps_per_epoch
        # other variables
        self.verbose = verbose
        self.validation_size = validation_size
        self.num_features = num_features
        self.random_seed = random_seed
        self.name = name
        self.dtype = dtype
        self.print_summary = print_summary
        self.query = query

    def _build_model(self):
        """
        TODO
        """
        # Placeholders for the inputs
        self.x0 = tf.keras.layers.Input(
            shape=self.num_features,
            dtype=self.dtype,
            name="x0"
        )

        self.x1 = tf.keras.layers.Input(
            shape=self.num_features,
            dtype=self.dtype,
            name="x1"
        )

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

        if self.drop_out > 0:
            nn = tf.keras.layers.Dropout(self.drop_out)(nn)

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

            if self.drop_out > 0:
                nn = tf.keras.layers.Dropout(self.drop_out)(nn)


        feature_part = tf.keras.models.Model(input_layer, nn, name='feature_part')

        if self.print_summary:
            feature_part.summary()

        nn0 = feature_part(self.x0)
        nn1 = feature_part(self.x1)

        subtracted = tf.keras.layers.Subtract()([nn0, nn1])

        out = tf.keras.layers.Dense(
            units=1,
            activation=self.ranking_activation_dr,
            use_bias=False,
            kernel_initializer=self.kernel_initializer_dr(seed=self.random_seed),
            kernel_regularizer=tf.keras.regularizers.l2(self.kernel_regularizer_dr),
            activity_regularizer=tf.keras.regularizers.l2(self.kernel_regularizer_dr),
            name="ranking_part"
        )(subtracted)

        self.model = tf.keras.models.Model(
            inputs=[(self.x0, self.x1)],
            outputs=out,
            name='Stacker'
        )

        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
            self.learning_rate,
            decay_steps=self.learning_rate_decay_steps,
            decay_rate=self.learning_rate_decay_rate,
            staircase=False
        )

        self.model.compile(
            optimizer=self.optimizer(lr_schedule),
            loss=self.loss,
            metrics=['acc']
        )

        if self.print_summary:
            self.model.summary()

    def fit(self, generator, **fit_params):
        """
        TODO
        """
        self._build_model()

        self.model.fit_generator(
            generator=generator,
            epochs=self.epoch,
            verbose=self.verbose,
            workers=1,
            callbacks=[PrintKendalTau(generator, pairwise=True), PrintTensor(generator, self.model)]
        )

    def predict_proba(self, features):
        """
        TODO
        """
        if len(features.shape) == 1:
            features = [features]

        res = self.model.predict([features, np.zeros(np.shape(features))], batch_size=self.batch_size, verbose=self.verbose)

        return res

    def predict(self, features, threshold):
        """
        TODO
        """
        if len(features.shape) == 1:
            features = [features]

        features_conv = np.expand_dims(features, axis=2)

        res = self.model.predict([features, np.zeros(np.shape(features))], batch_size=self.batch_size, verbose=self.verbose)

        return [1 if r > threshold else 0 for r in res]

    def make_pairs(self, x, y, x_l):
        x0_cur = []
        x1_cur = []
        for c in np.unique(y):
            if c == max(np.unique(y)): continue
            idx0 = np.random.randint(0, len(x_l[c + 1]), self.scale_factor_train_sample * len(x))
            idx1 = np.random.randint(0, len(x_l[c]), self.scale_factor_train_sample * len(x))
            x0_cur.extend(x_l[c + 1][idx0])
            x1_cur.extend(x_l[c][idx1])
        x0_cur = np.array(x0_cur)
        x1_cur = np.array(x1_cur)
        return x0_cur, x1_cur

    def make_pairs_query(self, x, y, q_ids, num_querys=0):
        x0_cur = []
        x1_cur = []
        if num_querys == 0:
            q_ids_selected = np.unique(q_ids)
        else:
            q_ids_selected = np.random.choice(np.unique(q_ids), num_querys, replace=False)
        for qi in q_ids_selected:
            x_q = x[q_ids == qi]
            y_q = y[q_ids == qi]
            sort_ids = np.argsort(y_q)
            x_q = x_q[sort_ids]
            y_q = y_q[sort_ids]
            if num_querys == 0:
                max_samples_qi = int(min(np.ceil(self.batch_size / len(q_ids_selected)), len(x_q)))
            else:
                max_samples_qi = len(x_q)
            # get random idxs between 0 and the query length - 1 (the last element can't ever be in the top position)
            idx0 = np.random.randint(0, len(x_q) - 1, max_samples_qi)
            # get random idxs between 1 and the value of idx0 at the same position. this is an offset instead of a idx
            idx1 = np.random.randint(1, len(y_q) - idx0, max_samples_qi)
            idx1 += idx0
            x0_cur.extend(x_q[idx0])
            x1_cur.extend(x_q[idx1])
        return np.array(x0_cur), np.array(x1_cur)



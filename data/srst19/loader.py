import tensorflow as tf
import pandas as pd
import numpy as np
from helpers import Constants

THIS_DIR = 'srst19'

def load_data(lang='en'):
    dir_loc = '{}/{}/{}/{}'.format(Constants.DATA_DIR, THIS_DIR, lang, lang)
    train_df = pd.read_csv('{}-train.H'.format(dir_loc), header=None)
    dev_df = pd.read_csv('{}-dev.H'.format(dir_loc), header=None)
    test_df = pd.read_csv('{}-test.H'.format(dir_loc), header=None)
    x_train, y_train, q_train = preprocess(train_df)
    x_dev, y_dev, q_dev = preprocess(dev_df)
    x_test, y_test, q_test = preprocess(test_df)
    return (x_train, y_train, q_train), (x_dev, y_dev, q_dev), (x_test, y_test, q_test)


def preprocess(df):
    y = df.iloc[:, -1]
    x = df.iloc[:, 1:-1]
    q = df.iloc[:, 0]
    return x.to_numpy(), y.to_numpy(), q.to_numpy()

class SRST19Generator(tf.keras.utils.Sequence):
        def __init__(self, batch_size=32, shuffle=True, language='en', query=True,
                     query_per_batch=5, split='train', pairwise=True):
            self.batch_size = batch_size
            self.train_data, self.dev_data, self.test_data = load_data(lang=language)
            self.q_order = np.random.permutation(range(1, max(self.train_data[2])-1))
            self.q_current = 0
            self.shuffle = shuffle
            self.query = query
            self.query_per_batch = query_per_batch
            self.current_split = split
            self.set_split(split)
            self.pairwise = pairwise
            self.on_epoch_end()

        def set_split(self, str):
            if str == 'train':
                self.data = self.train_data
            elif str == 'dev':
                self.data = self.dev_data
            elif str == 'test':
                self.data = self.test_data
            else:
                print('{} is not train/dev/test!'.format(str))
                exit(1)
            self.current_split = str

        def __len__(self):
            return int(np.floor(len(self.x[0]) / self.batch_size))

        def __getitem__(self, i):
            if self.query:
                if self.pairwise:
                    x0, x1, y = self.make_pairs_query()
                    return [x0, x1], y
                else:
                    return self.make_batch()
            else:
                return self.make_pairs(i)

        def make_pairs_query(self):
            q_ids_selected = np.random.choice(np.unique(self.q), self.query_per_batch, replace=False)
            x0_cur = []
            x1_cur = []
            for qi in q_ids_selected:
                q_idx = self.q == qi
                x_q = self.x[q_idx]
                y_q = self.y[q_idx]
                sort_ids = np.argsort(y_q)
                x_q = x_q[sort_ids]
                y_q = y_q[sort_ids]
                max_samples_qi = int(min(np.ceil(self.batch_size / self.query_per_batch), len(x_q)))
                # get random idxs between 0 and the query length - 1 (the last element can't ever be in the top position)
                idx0 = np.random.randint(0, len(x_q) - 1, max_samples_qi)
                # get random idxs between 1 and the value of idx0 at the same position. this is an offset instead of a idx
                idx1 = np.random.randint(1, len(y_q) - idx0, max_samples_qi)
                idx1 += idx0
                x0_cur.extend(x_q[idx0])
                x1_cur.extend(x_q[idx1])
                self.x = self.x[~q_idx]
                self.y = self.y[~q_idx]
                self.q = self.q[~q_idx]
            return np.array(x0_cur), np.array(x1_cur), np.ones(len(x0_cur))

        def make_batch(self):
            num_samples = 0
            x_cur = []
            y_cur = []
            while num_samples < self.batch_size:
                qi = self.q == self.q_order[self.q_current]
                x_q = self.x[qi]
                y_q = self.y[qi] / max(self.y[qi])
                x_cur.extend(x_q)
                y_cur.extend(y_q)
                self.q_current += 1
                num_samples = len(x_cur)
            return np.array(x_cur), np.array(y_cur)

        def make_batch_random(self):
            q_ids_selected = np.random.choice(np.unique(self.q), self.query_per_batch, replace=False)
            x_cur = []
            y_cur = []
            for qi in q_ids_selected:
                q_idx = self.q == qi
                x_q = self.x[q_idx]
                y_q = self.y[q_idx] / max(self.y[q_idx])
                self.x = self.x[~q_idx]
                self.y = self.y[~q_idx]
                self.q = self.q[~q_idx]
                x_cur.extend(x_q)
                y_cur.extend(y_q)
            return np.array(x_cur), np.array(y_cur)

        def make_pairs(self, i):
            pass

        def on_epoch_end(self):
            self.x, self.y, self.q = self.data
            self.q_current = 0

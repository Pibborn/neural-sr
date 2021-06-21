import tensorflow as tf
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from helpers import Constants

def preprocess(df):
    y = df.iloc[:, -1]
    x = df.iloc[:, 1:-1]
    q = df.iloc[:, 0]
    return x.to_numpy(), y.to_numpy(), q.to_numpy()

class DatasetGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataset, batch_size=32, shuffle=True, query=True, query_per_batch=10, split='train', pairwise=True, limit_dataset_size=None):
        self.limit_dataset_size = limit_dataset_size
        self.batch_size = batch_size
        self.dataset = dataset
        self.train_data, self.dev_data, self.test_data = self.load_data()
        self.q_order = list(range(1, max(self.train_data[2])))
        random.shuffle(self.q_order)
        self.q_current = 0
        self.shuffle = shuffle
        self.query = query
        self.query_per_batch = query_per_batch
        self.current_split = split
        self.set_split(split)
        self.pairwise = pairwise
        self.on_epoch_end()
        self.data_dict = self.prep_dict()

            
    def load_data(self, lang='en'):
        dir_loc = '{}/{}'.format(Constants.DATA_DIR, self.dataset)
        train_df = pd.read_csv(dir_loc + '-train.hvr', header=None, nrows=self.limit_dataset_size)
        dev_df = pd.read_csv(dir_loc + '-dev.hvr', header=None,
                             nrows=self.limit_dataset_size)
        test_df = pd.read_csv(dir_loc + '-test.hvr',
                              header=None, nrows=self.limit_dataset_size)
        x_train, y_train, q_train = preprocess(train_df)
        x_dev, y_dev, q_dev = preprocess(dev_df)
        x_test, y_test, q_test = preprocess(test_df)
        return (x_train, y_train, q_train), (x_dev, y_dev, q_dev), (x_test, y_test, q_test)

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
        if self.pairwise:
            return int(np.floor((len(np.unique(self.q)) - 1) / self.query_per_batch)) - 1
        else:
            return len(np.unique(self.q)) -1

    def __getitem__(self, i):
        if self.query:
            if self.pairwise:
                #x0, x1 = self.x0_epoch[i], self.x1_epoch[i]
                x0, x1 = self.make_pairs_query()
                y = np.ones(len(x0))
                return [x0, x1], y
            else:
                return self.make_batch_listnet()
        else:
            return self.make_pairs(i)

    def make_pairs_query(self):
        x0_cur = []
        x1_cur = []
        for i in range(self.query_per_batch):
            qi = self.q_order.pop()
            q_idx = self.q == qi
            x_q, y_q = self.data_dict[qi]
            if len(x_q) == 1:
                continue
            x_q = x_q[::-1]
            y_q = y_q[::-1]
            max_samples_qi = int(min(np.ceil(self.batch_size / self.query_per_batch), len(x_q)))
            # get random idxs between 0 and the query length - 1 (the last element can't ever be in the top position)
            idx0 = np.random.randint(0, len(x_q) - 1, max_samples_qi)
            # get random idxs between 1 and the value of idx0 at the same position. this is an offset instead of a idx
            idx1 = np.random.randint(1, len(y_q) - idx0, max_samples_qi)
            idx1 += idx0
            x0_cur.extend(x_q[idx0])
            x1_cur.extend(x_q[idx1])
        return np.array(x0_cur), np.array(x1_cur)

    def make_batch(self):
        num_samples = 0
        x_cur = []
        y_cur = []
        while num_samples < self.batch_size:
            qi = self.q == self.q_order[self.q_current]
            x_q = self.x[qi]
            y_q = 1 / self.y[qi]
            x_cur.extend(x_q)
            y_cur.extend(y_q)
            self.q_current += 1
            num_samples = len(x_cur)
        return np.array(x_cur), np.array(y_cur)

    def make_batch_listnet(self, i=None):
        if i is not None:
            q_current = i
        else:
            q_current = self.q_current
            self.q_current += 1
        qi = self.q == self.q_order[q_current]
        x_q = self.x[qi]
        assert len(x_q) > 0
        #y_q = np.argsort(self.y[qi])
        y_q = self.y[qi]
        return np.array(x_q), np.array(y_q, dtype=np.float32)

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
        self.q_current = 0
        self.q_order = list(range(1, max(self.train_data[2])))
        random.shuffle(self.q_order)
        self.x, self.y, self.q = self.data
        #if self.pairwise:
        #    self.x0_epoch = []
        #    self.x1_epoch = []
        #    for i in range(self.__len__()):
        #        if i % 2000 == 0:
        #            print('{}/{}'.format(i, self.__len__()))
        #        x0_i, x1_i = self.make_pairs_query()
        #        self.x0_epoch.append(x0_i)
        #        self.x1_epoch.append(x1_i)

    def prep_dict(self):
        data_dict = {}
        for qi in tqdm(self.q):
            q_idx = self.q == qi
            data_dict[qi] = (self.x[q_idx], self.y[q_idx])
        return data_dict


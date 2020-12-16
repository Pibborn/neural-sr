import pandas as pd
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

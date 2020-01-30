import numpy as np
import pandas as pd
import pickle
import os
import argparse

import surprise
from surprise import NormalPredictor
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate


parser = argparse.ArgumentParser(description='Hyperparameters')
parser.add_argument('--lr', '-l', default=0.001, type=float,
                    help='Learning rate')
parser.add_argument('--latent', '-v', default=10, type=int,
                    help='Latent vector size')
parser.add_argument('--seed', '-s', default=2017, type=int,
                    help='Random seed for numpy')

# Set hyperparameters and parameters
args = parser.parse_args()
lr = args.lr
latent_size = parser.latent
random_seed = args.seed

# Load files
TPS_DIR = 'data/toyandgame'

pkl_file = open(os.path.join(TPS_DIR, 'toyandgame.train'), 'rb')
train_data = pickle.load(pkl_file)
train_data = np.array(train_data)
pkl_file.close()
pkl_file = open(os.path.join(TPS_DIR, 'toyandgame.valid'), 'rb')
test_data = pickle.load(pkl_file)
test_data = np.array(test_data)
pkl_file.close()
print('Loading files done')

# Prepare training/validation sets
uid, iid, reuid, reiid, y_batch = zip(*train_data)
uid = np.array(uid)
iid = np.array(iid)
y_batch = np.array(y_batch)
df_train = pd.DataFrame({'userID': uid[:, 0], 'itemID': iid[:, 0], 'rating': y_batch[:, 0]})

uid_v, iid_v, reuid_v, reiid_v, y_batch_v = zip(*test_data)
uid_v = np.array(uid_v)
iid_v = np.array(iid_v)
y_batch_v = np.array(y_batch_v)
df_valid = pd.DataFrame({'userID': uid_v[:, 0], 'itemID': iid_v[:, 0], 'rating': y_batch_v[:, 0]})

reader = Reader(rating_scale=(1., 5.))
data_tr = Dataset.load_from_df(df_train[['userID', 'itemID', 'rating']], reader)
data_va = Dataset.load_from_df(df_valid[['userID', 'itemID', 'rating']], reader)

data_tr = data_tr.build_full_trainset()
data_va = data_va.build_full_trainset()
data_va = data_va.build_testset()
print('Training set: {} samples and validation set: {} samples prepared'.format(len(uid), len(uid_v)))


# Model training
algo=surprise.prediction_algorithms.matrix_factorization.NMF(n_factors=latent_size, lr_bu=lr, lr_bi=lr,\
                                                                 random_state=random_seed, biased=False)
algo.fit(data_tr)
pred_train = algo.test(data_tr.build_testset())
pred_valid = algo.test(data_va)
print('Train RMSE:', accuracy.rmse(pred_train))
print('Valid RMSE:', accuracy.rmse(pred_valid))

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
from utils import prepare_train_test_data


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


# Prepare training/validation sets
print('Loading files...')

TPS_DIR = 'data/toyandgame'
reader = Reader(rating_scale=(1., 5.))

data_tr, num_tr = prepare_train_test_data(TPS_DIR, 'toyandgame.train', reader)
data_va, num_va = prepare_train_test_data(TPS_DIR, 'toyandgame.valid', reader)
data_va = data_va.build_testset()

print('Training set: {} samples and validation set: {} samples prepared'.format(num_tr, num_va))


# Model training
algo = surprise.prediction_algorithms.matrix_factorization.NMF(n_factors=latent_size, lr_bu=lr, lr_bi=lr,
                                                               random_state=random_seed, biased=False)
algo.fit(data_tr)
pred_train = algo.test(data_tr.build_testset())
pred_valid = algo.test(data_va)
print('Train RMSE:', accuracy.rmse(pred_train))
print('Valid RMSE:', accuracy.rmse(pred_valid))

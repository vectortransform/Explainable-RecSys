import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

import pickle
import datetime
import argparse

import sys
sys.path.append('models/')

import tensorflow as tf
from tensorflow.keras.models import Model
from models import DeepRecSys
from utils import load_parameters, get_close_ids

parser = argparse.ArgumentParser(description='Input --lr or -l for learning rate tuning')
parser.add_argument('--lr', '-l', default=0.001, type=float,
                    help='Learning rate for Adam optimizer')
parser.add_argument('--reg', '-r', default=0.001, type=float, help='L2 regularizaion')
parser.add_argument('--dropout', '-d', default=0.5, type=float, help='Dropout keep probability')
parser.add_argument('--epoch', '-e', default=20, type=int, help='Number of epochs')
parser.add_argument('--seed', '-s', default=2017, type=int,
                    help='Random seed for numpy and tensorflow')

# Set hyperparameters and parameters
args = parser.parse_args()
lr = args.lr
l2_reg_lambda = args.reg
dropout_keep_prob = args.dropout
num_epochs = args.epoch
random_seed = args.seed

np.random.seed(random_seed)
tf.set_random_seed(random_seed)
batch_size = 100
embed_word_dim = 300
filter_size = 3
num_filters = 100

embed_id_dim = 32
attention_size = 32
n_latent = 32


# Load files
TPS_DIR = 'data/toyandgame'
print('Loadindg files...')

user_num, item_num, review_num_u, review_num_i, review_len_u, review_len_i,\
    vocabulary_user, vocabulary_item, train_length, test_length, u_text, i_text,\
    user_vocab_size, item_vocab_size = load_parameters(TPS_DIR, 'toyandgame.para')

initW_u = np.random.uniform(-1.0, 1.0, (user_vocab_size, 300))
initW_i = np.random.uniform(-1.0, 1.0, (user_vocab_size, 300))


# Build the model
model = DeepRecSys(l2_reg_lambda, random_seed, dropout_keep_prob, embed_word_dim, embed_id_dim,
                   filter_size, num_filters, attention_size, n_latent,
                   user_num, item_num, user_vocab_size, item_vocab_size,
                   review_num_u, review_len_u, review_num_i, review_len_i,
                   initW_u, initW_i, is_output_weights=True)

print('Model created with {} layers'.format(len(model.layers)))
print(model.summary())

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr, epsilon=1e-8),
              loss='mean_squared_error', metrics=['mse', 'mae'])

# Load trained model
model.load_weights('training_checkpoints/model_weights.hdf5')
print('Model weights loaded!')


# Laten representation of Users in attention layers
uid_emb = model.get_layer('user_id_embed').get_weights()[0]
uid_attn = np.matmul(uid_emb, model.get_layer('uesr_id_attention').get_weights()[0])


# tSNE visualization
uid_emb_2d = TSNE(n_components=2, verbose=1, n_jobs=-1).fit_transform(uid_emb)
plt.figure(figsize=(20, 20))
plt.plot(uid_emb_2d[:, 0], uid_emb_2d[:, 1], 'bo', markersize=5, alpha=.2)
plt.show()

uid_attn_2d = TSNE(n_components=2, verbose=1, n_jobs=-1).fit_transform(uid_attn)
uids = get_close_ids(uid_attn_2d)  # Pick two User IDs close to each other in the tSNE 2D space.
plt.figure(figsize=(20, 20))
plt.plot(uid_attn_2d[:, 0], uid_attn_2d[:, 1], 'bo', markersize=3, alpha=.1)
plt.plot(uid_attn_2d[uids, 0], uid_attn_2d[uids, 1],
         'ro', markerfacecolor='none', markersize=30, markeredgewidth=5)
plt.show()

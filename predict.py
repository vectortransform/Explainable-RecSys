import os
import numpy as np

import pickle
import datetime
import argparse

import sys
sys.path.append('models/')

import tensorflow as tf
from tensorflow.keras.models import Model
from models import *
from utils import load_parameters, load_train_test_data

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

uid_te, iid_te, reuid_te, reiid_te, yrating_te, texts_u_te, texts_i_te = load_train_test_data(
    TPS_DIR, 'toyandgame.PredictRequest', u_text, i_text)

initW_u = np.random.uniform(-1.0, 1.0, (user_vocab_size, 300))
initW_i = np.random.uniform(-1.0, 1.0, (user_vocab_size, 300))

print('Test set: {} samples prepared'.format(len(uid_te)))


# Build the model
input_u = Input(shape=(review_num_u, review_len_u), dtype='int32', name='texts_u')
input_i = Input(shape=(review_num_i, review_len_i), dtype='int32', name='texts_i')
input_uid = Input(shape=(1), dtype='int32', name='uid')
input_iid = Input(shape=(1), dtype='int32', name='iid')

model = DeepRecSys(input_u, input_i, input_uid, input_iid,
                   l2_reg_lambda, random_seed, dropout_keep_prob, embed_word_dim, embed_id_dim,
                   filter_size, num_filters, attention_size, n_latent,
                   user_num, item_num, user_vocab_size, item_vocab_size,
                   review_num_u, review_len_u, review_num_i, review_len_i,
                   initW_u, initW_i)

print('Model created with {} layers'.format(len(model.layers)))
print(model.summary())

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr, epsilon=1e-8),
              loss='mean_squared_error', metrics=['mse', 'mae'])

# Load trained model
model.load_weights('training_checkpoints/model_weights.hdf5')
print('Model weights loaded!')

time_str = datetime.datetime.now().isoformat()
print(time_str)

# Rating prediction
print('Prediction starts!')
ratings = model.predict(
    {'texts_u': texts_u_te, 'texts_i': texts_i_te, 'uid': uid_te, 'iid': iid_te},
    batch_size=batch_size,
)

# Review-usefulness prediction
layer_i = model.layers[31]
activation_i = Model(inputs=model.input, outputs=layer_i.output)
review_weights = activation_i.predict(
    {'texts_u': texts_u_te, 'texts_i': texts_i_te, 'uid': uid_te, 'iid': iid_te},
    batch_size=batch_size
)

print(ratings)
print(review_weights)

print('Prediction complete')
time_str = datetime.datetime.now().isoformat()
print(time_str)

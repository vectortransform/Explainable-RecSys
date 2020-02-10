import os
import numpy as np

import pickle
import datetime
import argparse

import sys
sys.path.append('models/')

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from models import DeepRecSys
from utils import load_parameters, load_train_test_data, load_word_embedding_weights


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

uid_tr, iid_tr, reuid_tr, reiid_tr, yrating_tr, texts_u_tr, texts_i_tr = load_train_test_data(
    TPS_DIR, 'toyandgame.train', u_text, i_text)
uid_va, iid_va, reuid_va, reiid_va, yrating_va, texts_u_va, texts_i_va = load_train_test_data(
    TPS_DIR, 'toyandgame.valid', u_text, i_text)
print('Training set: {} samples and validation set: {} samples prepared'.format(len(uid_tr), len(uid_va)))

initW_u, initW_i = load_word_embedding_weights(TPS_DIR, 'initW_u', 'initW_i')
print('word2vec weights initialization done')


# Build the model
model = DeepRecSys(l2_reg_lambda, random_seed, dropout_keep_prob, embed_word_dim, embed_id_dim,
                   filter_size, num_filters, attention_size, n_latent,
                   user_num, item_num, user_vocab_size, item_vocab_size,
                   review_num_u, review_len_u, review_num_i, review_len_i,
                   initW_u, initW_i)

print('Model created with {} layers'.format(len(model.layers)))
print(model.summary())

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr, epsilon=1e-8),
              loss='mean_squared_error', metrics=['mse', 'mae'])


checkpoint_dir = 'training_checkpoints_{}'.format(lr)
os.mkdir(checkpoint_dir)
checkpoint_prefix = os.path.join(checkpoint_dir, "{epoch:02d}-{val_loss:.3f}.hdf5")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix, save_weights_only=True)


# Training
print('Training begins with lr={}, l2_reg_lambda={}, dropout_keep={}, num_epochs={}, batch_size={}'.format(
    lr, l2_reg_lambda, dropout_keep_prob, num_epochs, batch_size))
time_str = datetime.datetime.now().isoformat()
print(time_str)

history_fit = model.fit(
    {'texts_u': texts_u_tr, 'texts_i': texts_i_tr, 'uid': uid_tr, 'iid': iid_tr},
    yrating_tr,
    validation_data=({'texts_u': texts_u_va, 'texts_i': texts_i_va, 'uid': uid_va, 'iid': iid_va},
                     yrating_va),
    epochs=num_epochs,
    batch_size=100,
    verbose=0,
    callbacks=[checkpoint_callback]
)


# Save the training history
pickle.dump(history_fit.history, open('lr{}_history_fit'.format(lr), 'wb'))

print('Training complete')
time_str = datetime.datetime.now().isoformat()
print(time_str)

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

f_para = open(os.path.join(TPS_DIR, 'toyandgame.para'), 'rb')
para = pickle.load(f_para)
user_num = para['user_num']
item_num = para['item_num']
review_num_u = para['review_num_u']
review_num_i = para['review_num_i']
review_len_u = para['review_len_u']
review_len_i = para['review_len_i']
vocabulary_user = para['user_vocab']
vocabulary_item = para['item_vocab']
train_length = para['train_length']
test_length = para['test_length']
u_text = para['u_text']
i_text = para['i_text']
user_vocab_size = len(vocabulary_user)
item_vocab_size = len(vocabulary_item)
f_para.close()

pkl_file = open(os.path.join(TPS_DIR, 'toyandgame.PredictRequest'), 'rb')
test_data = pickle.load(pkl_file)
test_data = np.array(test_data)
pkl_file.close()

initW_u = np.random.uniform(-1.0, 1.0, (user_vocab_size, 300))
initW_i = np.random.uniform(-1.0, 1.0, (user_vocab_size, 300))
print('Loading files done')

uid_v, iid_v, reuid_v, reiid_v, y_batch_v = zip(*test_data)
uid_v = np.array(uid_v)
iid_v = np.array(iid_v)
reuid_v = np.array(reuid_v)
reiid_v = np.array(reiid_v)
y_batch_v = np.array(y_batch_v)

texts_u_v=[]
texts_i_v=[]
for i in range(len(uid_v)):
    texts_u_v.append(u_text[uid_v[i][0]])
    texts_i_v.append(i_text[iid_v[i][0]])
texts_u_v = np.array(texts_u_v)
texts_i_v = np.array(texts_i_v)

print('Test set: {} samples prepared'.format(len(uid_v)))


# Build the model
input_u = Input(shape=(review_num_u, review_len_u), dtype='int32', name='texts_u')
input_i = Input(shape=(review_num_i, review_len_i), dtype='int32', name='texts_i')
input_reuid = Input(shape=(review_num_u), dtype='int32', name='reuid')
input_reiid = Input(shape=(review_num_i), dtype='int32', name='reiid')
input_uid = Input(shape=(1), dtype='int32', name='uid')
input_iid = Input(shape=(1), dtype='int32', name='iid')

model = DeepRecSys(input_u, input_i, input_reuid, input_reiid, input_uid, input_iid,\
                   l2_reg_lambda, random_seed, dropout_keep_prob, embed_word_dim, embed_id_dim,\
                   filter_size, num_filters, attention_size, n_latent,\
                   user_num, item_num, user_vocab_size, item_vocab_size,\
                   review_num_u, review_len_u, review_num_i, review_len_i,\
                   initW_u, initW_i)

print('Model created with {} layers'.format(len(model.layers)))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr, epsilon=1e-8), loss='mean_squared_error', metrics=['mse', 'mae'])

# Load trained model
model.load_weights('training_checkpoints/model_weights.hdf5')
print('Model weights loaded!')

time_str = datetime.datetime.now().isoformat()
print(time_str)

# Rating prediction
ratings = model.predict(
    {'texts_u': texts_u, 'texts_i': texts_i, 'reuid': reuid, 'reiid': reiid, 'uid': uid, 'iid':iid},
     batch_size=batch_size,
     )

# Review-usefulness prediction
layer_i = model.layers[29]
activation_i = Model(inputs=model.input, outputs=layer_i.output)
review_weights = activation_i.predict(
    {'texts_u': texts_u_v, 'texts_i': texts_i_v, 'reuid': reuid_v, 'reiid': reiid_v, 'uid': uid_v, 'iid':iid_v},
    batch_size=batch_size
)

print(ratings)
print(review_weights)

print('Prediction complete')
time_str = datetime.datetime.now().isoformat()
print(time_str)

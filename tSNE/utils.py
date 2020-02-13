import os
import numpy as np
from scipy.spatial import distance_matrix
import pickle


def load_parameters(TPS_DIR, filename):
    """
    Load the parameters for the model

    Args:
    -------
    TPS_DIR: directory of the data folder
    filename: file name

    Outputs:
    -------
    : parameters for building the model
    """

    f_para = open(os.path.join(TPS_DIR, filename), 'rb')
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

    return user_num, item_num, review_num_u, review_num_i, review_len_u, review_len_i,\
        vocabulary_user, vocabulary_item, train_length, test_length, u_text, i_text,\
        user_vocab_size, item_vocab_size


def get_close_ids(uid_attn_2d):
    """
    Pick two User IDs close to each other in the tSNE 2D space.

    Args:
    -------
    uid_attn_2d: t-SNE 2D representation of users in the attention layers

    Outputs:
    -------
    : two closest User IDs
    """

    dis_mat = distance_matrix(uid_attn_2d, uid_attn_2d)
    for i in range(len(dis_mat)):
        dis_mat[i][i] = np.inf
    indices = np.unravel_index(np.argmin(dis_mat, axis=None), dis_mat.shape)

    return indices

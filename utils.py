import os
import numpy as np
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


def load_train_test_data(TPS_DIR, filename, u_text, i_text):
    """
    Prepare training/validation/test sets

    Args:
    -------
    TPS_DIR: directory of the data folder
    filename: file name of the dataset
    u_text: dictionary for user's reviews
    i_text: dictionary for item's reviews

    Outputs:
    -------
    : training/validation/test sets
    """

    pkl_file = open(os.path.join(TPS_DIR, filename), 'rb')
    data = pickle.load(pkl_file)
    data = np.array(data)
    pkl_file.close()

    uid, iid, reuid, reiid, y_batch = zip(*data)
    uid = np.array(uid)
    iid = np.array(iid)
    reuid = np.array(reuid)
    reiid = np.array(reiid)
    y_batch = np.array(y_batch)

    texts_u = []
    texts_i = []
    for i in range(len(uid)):
        texts_u.append(u_text[uid[i][0]])
        texts_i.append(i_text[iid[i][0]])
    texts_u = np.array(texts_u)
    texts_i = np.array(texts_i)

    return uid, iid, reuid, reiid, y_batch, texts_u, texts_i


def load_word_embedding_weights(TPS_DIR, filename_u, filename_i):
    """
    Load pretrained wording embedding weights for the model

    Args:
    -------
    TPS_DIR: directory of the data folder
    filename_u: file name for the embededding matrix for user's reviews
    filename_i: file name for the embededding matrix for item's reviews

    Outputs:
    -------
    : user and item word embedding
    """

    pkl_file = open(os.path.join(TPS_DIR, filename_u), 'rb')
    initW_u = pickle.load(pkl_file)
    pkl_file.close()
    pkl_file = open(os.path.join(TPS_DIR, filename_i), 'rb')
    initW_i = pickle.load(pkl_file)
    pkl_file.close()

    return initW_u, initW_i

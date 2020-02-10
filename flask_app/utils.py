import os
import numpy as np
import pickle


def load_parameters(TPS_DIR, filename):
    # Load the parameters for the model
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
    # Load and prepare training/validation/test sets
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

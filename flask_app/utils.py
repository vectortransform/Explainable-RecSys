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


def tf_serving(texts_u, texts_i, user_ids, item_ids):
    """
    Post the inputs to TensorFlow Serving model and obtain predictions

    Args:
    -------
    texts_u: users' reviews
    texts_i: items' reviews
    user_ids: user ids
    item_ids: item ids

    Outputs:
    -------
    : model outputs for rating/ranking prediction
    : inference time
    """

    insts = {'texts_u:0': texts_u, 'texts_i:0': texts_i,
             'uid:0': user_ids.tolist(), 'iid:0': item_ids.tolist()}
    data = json.dumps({"signature_name": "serving_default", "inputs": insts})
    headers = {"content-type": "application/json"}
    time_a = datetime.datetime.now()
    json_response = requests.post(
        'http://localhost:8501/v1/models/recsys_model:predict', data=data, headers=headers)
    time_b = datetime.datetime.now()
    time_dif = time_b - time_a
    res = json.loads(json_response.text)['outputs']

    return res, time_dif


def get_metadata(df_meta, item_ids_new, num_top=10, single_pred=False):
    """
    Prepare the metadata for top 10 suggested items

    Args:
    -------
    df_meta: Pandas DataFrame of metadata
    item_ids_new: sorted item ids based on their rating prediction
    num_top: number of top items

    Outputs:
    -------
    : item metadata of description, title, price, image url and categories
    """

    if single_pred:
        sample = df_meta.loc[item_ids_new]
        if sample['asin'] != item_id:
            print('Wrong id metadata', item_id, sample['asin'])
        else:
            des_meta = sample['description']
            title_meta = sample['title']
            price_meta = sample['price']
            imurl_meta = sample['imUrl']
            categ_meta = sample['categories']

    else:
        des_meta = []
        title_meta = []
        price_meta = []
        imurl_meta = []
        categ_meta = []

        for i in range(num_top):
            sample = df_meta.loc[item_ids_new[i]]
            if sample['asin'] != item_ids_new[i]:
                print('Wrong id metadata', item_ids_new[i], sample['asin'])
            else:
                des_meta.append(sample['description'])
                title_meta.append(sample['title'])
                price_meta.append(sample['price'])
                imurl_meta.append(sample['imUrl'])
                categ_meta.append(sample['categories'])

    return des_meta, title_meta, price_meta, imurl_meta, categ_meta

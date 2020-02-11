import os
import pickle
import datetime

import numpy as np
import pandas as pd
import json
import requests
from flask import Flask, request, jsonify, render_template

from utils import load_parameters, tf_serving, get_metadata

# Read files
np.random.seed(2017)
TPS_DIR = '/home/ubuntu/data/toyandgame'
print('Loadindg data...')

user_num, item_num, review_num_u, review_num_i, review_len_u, review_len_i,\
    vocabulary_user, vocabulary_item, train_length, test_length, u_text, i_text,\
    user_vocab_size, item_vocab_size = load_parameters(TPS_DIR, 'toyandgame.para')

# item ratings from reviews
pkl_file = open(os.path.join(TPS_DIR, 'df_revrate'), 'rb')
df_revrate = pickle.load(pkl_file)
pkl_file.close()

# meta data for items
pkl_file = open(os.path.join(TPS_DIR, 'df_meta'), 'rb')
df_meta = pickle.load(pkl_file)
pkl_file.close()

# user's review texts
pkl_file = open(os.path.join(TPS_DIR, 'user_rev_original'), 'rb')
user_rev_original = pickle.load(pkl_file)
pkl_file.close()

# item's review texts
pkl_file = open(os.path.join(TPS_DIR, 'item_rev_original'), 'rb')
item_rev_original = pickle.load(pkl_file)
pkl_file.close()
print('Loadindg data done')


# Flask server
app = Flask(__name__)


@app.route('/hello/', methods=['GET', 'POST'])
def hello_world():
    """
    Test the Flask App
    """

    return 'Hello, World!'


@app.route('/', methods=['POST', 'GET'])
def index():
    """
    Index page
    """

    if request.method == 'POST':
        return render_template('index.html')

    else:
        return render_template('index.html')


@app.route('/candidate', methods=['GET', 'POST'])
def generate_candidate():
    """
    Generate 100 items as candidates
    """

    if request.method == 'POST':
        user_id = int(request.form['uid'])
        item_ids = np.random.randint(0, item_num, size=100)

        return render_template('candidate.html', result=item_ids.tolist(), last_uid=user_id)

    else:
        return render_template('candidate.html')


@app.route('/ranking', methods=['GET', 'POST'])
def ranking_item():
    """
    Rank the items based on their predicted ratings

    Outputs:
    -------
    : predicted ratings, inference time, ordered item ids, item metadata
    """

    if request.method == 'POST':
        ids = request.json
        user_id = int(ids['uid'])
        item_ids = ids['iids']

        user_ids = np.full(100, user_id)
        item_ids = item_ids[1:-1].split(',')
        item_ids = np.array(item_ids).astype(int)

        texts_u = []
        texts_i = []
        for i in user_ids:
            texts_u.append(u_text[i].tolist())
        for j in item_ids:
            texts_i.append(i_text[j].tolist())

        user_ids = user_ids.reshape(-1, 1)
        item_ids = item_ids.reshape(-1, 1)

        # Feed the inputs to the Tensorflow Serving model
        res, time_dif = tf_serving(texts_u, texts_i, user_ids, item_ids)

        # Get the ranking results
        rating = np.array(res['final_rating/add_1:0']).reshape(-1)
        order = np.argsort(rating)[::-1]
        item_ids_new = item_ids.reshape(-1)[order]
        rating_new = rating[order]

        # Prepare the metadata for 10 suggested items
        des_meta, title_meta, price_meta, imurl_meta, categ_meta = get_metadata(
            df_meta, item_ids_new, num_top=10)

        return json.dumps({'rating': rating_new.tolist(), 'infertime': time_dif.total_seconds(), 'item_ids': item_ids_new.tolist(),
                           'des_meta': des_meta, 'title_meta': title_meta, 'price_meta': price_meta, 'imurl_meta': imurl_meta, 'categ_meta': categ_meta})

    else:
        return render_template('candidate.html')


@app.route('/predictreview', methods=['GET', 'POST'])
def rating_review():
    """
    Predict personalized review-usefulness

    Outputs:
    -------
    : predicted ratings, inference time, top reviews with ratings, other reviews with ratings, item metadata
    """
    if request.method == 'POST':
        ids = request.json
        user_id = int(ids['uid'])
        item_id = int(ids['iid'])

        # Feed the inputs to the Tensorflow Serving model
        res, time_dif = tf_serving([u_text[user_id].tolist()], [i_text[item_id].tolist()],
                                   np.array([[user_id]]), np.array([[item_id]]))

        # Get the rating and ordered reviews based on their review-usefulness
        rating = np.array(res['final_rating/add_1:0']).reshape(-1)
        item_rev_weights = np.array(res['item_rev_weights/transpose_1:0']).reshape(-1)

        order = np.argsort(item_rev_weights)[::-1]
        rev_texts = item_rev_original[item_id][:review_num_i]
        if len(rev_texts) < review_num_i:
            rev_texts = rev_texts + [''] * (review_num_i - len(rev_texts))
        rev_texts = np.array(rev_texts)[order]

        # Top-3 reviews and other reviews
        toprevs = []
        otherrevs = []

        for i, rev_text in enumerate(rev_texts):
            if rev_text:
                if i < 3 or len(toprevs) < 3:
                    toprevs.append(rev_text)
                else:
                    otherrevs.append(rev_text)

        rev_rate_top = [int(float(df_revrate[toprev])) for toprev in toprevs]
        rev_rate_other = [int(float(df_revrate[otherrev])) for otherrev in otherrevs]

        # Prepare the metadata for the item
        des_meta, title_meta, price_meta, imurl_meta, categ_meta = get_metadata(
            df_meta, item_id, single_pred=True)

        return json.dumps({'rating': rating.tolist(), 'infertime': time_dif.total_seconds(),
                           'toprevs': toprevs, 'otherrevs': otherrevs, 'rev_rate_top': rev_rate_top, 'rev_rate_other': rev_rate_other,
                           'des_meta': des_meta, 'title_meta': title_meta, 'price_meta': price_meta, 'imurl_meta': imurl_meta, 'categ_meta': categ_meta})

    else:
        return render_template('candidate.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

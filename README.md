# Explainable RecSys
Explainable RecSys is a personalized, review-level explainable recommendation system that can not only surface relevant items to users, but also highlight product reviews most relevant to their personal preferences simultaneously.

<img src="https://github.com/CoserU/Explainable-RecSys/blob/master/figures/webapp.gif">

## Overview
In current e-commerce and review-based applications, after the recommendation systems suggest an item to a prospective user, they often leave it to the customer to sift through numerous item’s reviews of varying quality and relevance. Moreover, majority of the reviews, whether labeled as helpful or not, are not very useful to the specific customer, because they were written by other people and different people focus on different product aspects.

I developed an attention-based DNN architecture which is capable to learn personalized review-usefulness even without "this review has been labeled as helpful" information, which is absent in many online systems. My major modification to the existing models is designing the matching score function so that the user embedding can interact with each of the item reviews to learn their usefulness for this customer. The review-level explanation makes the system become more transparent and helps increase customers’ trust and satisfaction.

<img src="https://github.com/CoserU/Explainable-RecSys/blob/master/figures/DNN_architecture.png">

## Environment
Python 3.6

TensorFlow 1.15

The model was trained and served on AWS EC2 (Deep Learning AMI) with `tensorflow_p36` conda environment. To create the same environment, one can use the following command:

`conda env create -f tensorflow_p36.yml`

## Model Training
Prepare the training/test sets via `preprocessing/data.ipynb` and train the model by running `train.py`. You can find the code for the attention-based DNN architecture in `models/deeprecsys.py`.

## Model Serving
Save the model via `tensorflow_serving.py` and deploy the Flask-based web application by running `flask_app/run.sh`.

<img src="https://github.com/CoserU/Explainable-RecSys/blob/master/figures/ML_pipeline.png">

## User Modeling (Future Work)
The Explainable RecSys is also capable to model user behaviors/preferences:
1. Identify customers who always write helpful reviews (e.g., Amazon Vine).
2. Group users who prefer certain product aspects.

Below shows tSNE visualization of latent representation of users before the attention layers (left) and in the attention layers (right). You can find the code in `tSNE/tSNE_visualization.py`.

<table><tr>
  <td> <img src="https://github.com/CoserU/Explainable-RecSys/blob/master/figures/tSNE_before_attention.png" /> </td>
  <td> <img src="https://github.com/CoserU/Explainable-RecSys/blob/master/figures/tSNE_in_attention.png" /> </td>
</tr></table>
    
Apparently, the attention layers have learned these nice user groups/clustering here, compared with their inputs. If we pick two neighbour users close to each other in the right figure and track their review history, we can see that both of them focus on the similar product aspects, e.g., toy lifetime.

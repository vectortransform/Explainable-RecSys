# Explainable RecSys
Explainable RecSys is a personalized, review-level explainable recommendation system that can not only surface relevant items to users, but also highlight product reviews most relevant to their personal preferences simultaneously.

## Environment
Python 3.6

TensorFlow 1.15

The model was trained and served on AWS EC2 (Deep Learning AMI) with `tensorflow_p36` conda environment. To create the same environment, one can use the following command:

`conda env create -f tensorflow_p36.yml`

## Model Training
Prepare the training/test sets via `preprocessing/data.ipynb` and train the model by running `train.py`. You can find the novel attention-based DNN architecture in `models`.

## Model Serving
Save the model via `tensorflow_serving.py` and deploy the Flask-based web application by running `flask_app/run.sh`.

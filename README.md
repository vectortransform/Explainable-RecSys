# Explainable RecSys
The deep learning-based Explainable RecSys is a personalized, explainable recommendation system that surfaces relevant items to users while simultaneously highlighting reviews most relevant to their personal preferences.

## Environment
Python 3.6

TensorFlow 1.15

The model was trained and served on AWS EC2 (Deep Learning AMI) with `tensorflow_p36` conda environment. To create the same environment, one can use the following command:

`conda env create -f tensorflow_p36.yml`

## Model Training
Prepare the training/test sets via `preprocessing/data.ipynb` and train the model by running `train.py`.

## Model Serving
Save the model via `tensorflow_serving.py` and deploy the Flask-based web application by running `flask_app/run.sh`.

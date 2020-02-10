#!/bin/sh

nohup tensorflow_model_server \
  --rest_api_port=8501 \
  --model_name=recsys_model \
  --model_base_path=/home/ubuntu/model_serve/ >server.log 2>&1 &

sleep 5

nohup python app.py &

#!/usr/bin/env bash

INSTANCE=mnist-node1
USER=rablanken13

cd ../

init() {
    gcloud compute ssh --command 'sudo apt-get install -y python-pip' ${USER}@${INSTANCE}
    gcloud compute ssh --command 'mkdir -p data' ${USER}@${INSTANCE}
    gcloud compute ssh --command 'mkdir -p mnist_models' ${USER}@${INSTANCE}
    gcloud compute scp --compress data/train.csv ${USER}@${INSTANCE}:data
    gcloud compute scp --compress data/test.csv ${USER}@${INSTANCE}:data
    gcloud compute scp requirements.txt ${USER}@${INSTANCE}:
    gcloud compute ssh --command 'pip install -r requirements.txt' ${USER}@${INSTANCE}
}

deploy() {
    gcloud compute scp main.py ${USER}@${INSTANCE}:
    gcloud compute scp mnist_data_csv.py ${USER}@${INSTANCE}:
    gcloud compute scp mnist_models/__init__.py ${USER}@${INSTANCE}:mnist_models/
    gcloud compute scp mnist_models/conv_2d_v2.py ${USER}@${INSTANCE}:mnist_models/
}

run() {
    gcloud compute ssh --command 'python main.py' ${USER}@${INSTANCE}
}

#init
deploy
run

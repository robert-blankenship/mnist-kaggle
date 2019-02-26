#!/usr/bin/env bash


startup() {
    gcloud compute instances create mnist-node1 \
        --preemptible --tags mnist \
        --scopes cloud-platform
    gcloud compute instances start mnist-node1

#    gcloud alpha cloud-shell c
    # would like a way to start a TPU
}

startup
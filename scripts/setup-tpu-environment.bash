#!/usr/bin/env bash


startup() {
    gcloud compute instances create mnist-node1 \
        --preemptible --tags mnist \
        --scopes cloud-platform
}

startup
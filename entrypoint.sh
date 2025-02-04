#!/bin/bash

# Copyright (c) 2025 The Trustees of the University of Illinois
# licensed under the MIT License
# see "LICENSE" file in the distribution root directory

set -e
cd $(dirname $0)

if [ $1 == "shell" ]; then
    shift
    exec /bin/bash $@
else
    if [ -e /usr/bin/nvidia-smi ]; then
        echo "GPUs"
        /usr/bin/nvidia-smi -L
    fi
    echo "Use shell to start a shell in the container"
    exec python pipeline.py $@
fi

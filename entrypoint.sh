#!/bin/bash

set -e
cd $(dirname $0)

if [ $1 == "shell" ]; then
    shift
    exec /bin/bash $@
else
    echo "Use shell to start a shell in the container"
    exec python pipeline.py $@
fi

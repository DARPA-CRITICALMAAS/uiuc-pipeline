#!/bin/bash

set -e
cd $(dirname $0)

# default values
MODEL="${MODEL:-primordial_positron_3}"
FEATURE_TYPE="${FEATURE_TYPE:-polygon}"
DATA_FOLDER="${DATA_FOLDER:-/data}"
OUTPUT_FOLDER="${OUTPUT_FOLDER:-/output}"
if [ -d /legends ]; then
    LEGENDS_FOLDER="${LEGENDS_FOLDER:-/legends}"    
fi
if [ -d /layouts ]; then
    LAYOUTS_FOLDER="${LAYOUTS_FOLDER:-/layouts}"
fi
if [ -d /validation ]; then
    VALIDATION_FOLDER="${VALIDATION_FOLDER:-/validation}"
fi
if [ -d /feedback ]; then
    FEEDBACK_FOLDER="${FEEDBACK_FOLDER:-/feedback}"
fi
if [ -d /checkpoints ]; then
    CHECKPOINTS_FOLDER=${CHECKPOINTS_FOLDER:-/checkpoints}
fi

# parse arguments
ARGS=""
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --layouts)
            if [ ! -d $2 ]; then
                echo "Layouts folder does not exist: $2"
                exit 1
            fi
            LAYOUTS_FOLDER="$2"
            shift
            shift
            ;;
        --data)
            if [ ! -d $2 ]; then
                echo "Data folder does not exist: $2"
                exit 1
            fi
            DATA_FOLDER="$2"
            shift
            shift
            ;;
        --feature_type)
            FEATURE_TYPE="$2"
            shift
            shift
            ;;
        --feedback)
            if [ ! -d $2 ]; then
                echo "Feedback folder does not exist: $2"
                exit 1
            fi
            FEEDBACK_FOLDER="$2"
            shift
            shift
            ;;
        --legends)
            if [ ! -d $2 ]; then
                echo "Legends folder does not exist: $2"
                exit 1
            fi
            LEGENDS_FOLDER="$2"
            shift
            shift
            ;;
        --log)
            ARGS="${ARGS} --log ${2}"
            shift
            shift
            ;;
        --model)
            MODEL="$2"
            shift
            shift
            ;;
        --output)
            if [ ! -d $2 ]; then
                echo "Output folder does not exist: $2"
                exit 1
            fi
            OUTPUT_FOLDER="$2"
            shift
            shift
            ;;
        --validation)
            if [ ! -d $2 ]; then
                echo "Validation folder does not exist: $2"
                exit 1
            fi
            VALIDATION_FOLDER="$2"
            shift
            shift
            ;;
        -v|--verbose)
            ARGS="${ARGS} --verbose"
            shift
            ;;
        -s|--shell)
            exec /bin/bash
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# sanity checks
if [ ! -d ${DATA_FOLDER} ]; then
    echo "Data folder does not exist: ${DATA_FOLDER}"
    exit 1
fi
if [ ! -d ${OUTPUT_FOLDER} ]; then
    echo "Output folder does not exist: ${OUTPUT_FOLDER}"
    exit 1
fi

# create arguments
ARGS="${ARGS} --model ${MODEL} --feature_type ${FEATURE_TYPE} --data ${DATA_FOLDER} --output ${OUTPUT_FOLDER}"
if [ -n "${LEGENDS_FOLDER}" ]; then
    ARGS="${ARGS} --legends ${LEGENDS_FOLDER}"
fi
if [ -n "${LAYOUTS_FOLDER}" ]; then
    ARGS="${ARGS} --layouts ${LAYOUTS_FOLDER}"
fi
if [ -n "${VALIDATION_FOLDER}" ]; then
    ARGS="${ARGS} --validation ${VALIDATION_FOLDER}"
fi
if [ -n "${FEEDBACK_FOLDER}" ]; then
    ARGS="${ARGS} --feedback ${FEEDBACK_FOLDER}"
fi

# execute pipeline
echo ${ARGS}
exec python pipeline.py ${ARGS}

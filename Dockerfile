FROM --platform=linux/amd64 python:3.9

# setup packages needed
RUN apt-get update && \
    apt-get -y install python3-gdal libgdal-dev libgl1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p /projects/bbym/shared && \
    ln -s /src/checkpoints /projects/bbym/shared/models

# install requirements
COPY requirements.txt /src/requirements.txt
RUN pip install --no-cache -r /src/requirements.txt

# setup folders
WORKDIR /src
VOLUME /data, /output, /legends, /layouts, /validation, /feedback, /checkpoints

# setup environment variables
ENV MODEL="golden_muscat" \
    FEATURE_TYPE="polygon" \
    DATA_FOLDER="" \
    OUTPUT_FOLDER="" \
    LEGENDS_FOLDER="" \
    LAYOUTS_FOLDER="" \
    VALIDATION_FOLDER="" \
    FEEDBACK_FOLDER="" \
    CHECKPOINTS_FOLDER="" \
    AMQP=""

# install application
COPY . /src/

# run application
ENTRYPOINT [ "/src/entrypoint.sh" ]
CMD "--help"

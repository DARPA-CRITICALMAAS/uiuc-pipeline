FROM --platform=linux/amd64 python:3.9

# setup folders
WORKDIR /src
VOLUME /data, /output, /legends, /layouts, /validation, /feedback, /checkpoints

# setup environment variables
ENV MODEL="golden_muscat" \
    FEATURE_TYPE="polygon" \
    DATA_FOLDER="/data" \
    OUTPUT_FOLDER="/output" \
    LEGENDS_FOLDER="" \
    LAYOUTS_FOLDER="" \
    VALIDATION_FOLDER="" \
    FEEDBACK_FOLDER="" \
    CHECKPOINTS_FOLDER=""

# setup packages needed
RUN apt-get update && \
    apt-get -y install python3-gdal libgdal-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p /projects/bbym/shared && \
    ln -s /src/checkpoints /projects/bbym/shared/models

# install requirements
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache -r /app/requirements.txt

# install application
COPY . /src/

# run application
ENTRYPOINT [ "/src/entrypoint.sh" ]
CMD ["-v"]

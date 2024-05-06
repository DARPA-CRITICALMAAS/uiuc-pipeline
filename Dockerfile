FROM --platform=linux/amd64 python:3.10

# setup packages needed
RUN apt-get update && \
    apt-get -y install python3-gdal libgdal-dev libgl1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p /projects/bbym/shared && \
    ln -s /src/checkpoints /projects/bbym/shared/models

# install requirements, skip all repos starting with -e
COPY requirements.txt /src/requirements.txt
RUN sed -i 's#^\(-e .*\)$#\#\1#' /src/requirements.txt && \
    pip install --no-cache -r /src/requirements.txt

# setup folders
WORKDIR /src
VOLUME /data, /output, /legends, /layouts, /validation, /feedback, /checkpoints

# setup environment variables
ARG CDR_SYSTEM_VERSION="unknown"
ENV MODEL="golden_muscat" \
    FEATURE_TYPE="polygon" \
    DATA_FOLDER="" \
    OUTPUT_FOLDER="" \
    LEGENDS_FOLDER="" \
    LAYOUTS_FOLDER="" \
    VALIDATION_FOLDER="" \
    FEEDBACK_FOLDER="" \
    CHECKPOINTS_FOLDER="" \
    CDR_SYSTEM="UIUC" \
    CDR_SYSTEM_VERSION="${CDR_SYSTEM_VERSION}" \
    AMQP=""

# install application
COPY . /src/

# reinstall requirements to install those we skipped earlier
RUN pip install --no-cache -r /src/requirements.txt

# run application
ENTRYPOINT [ "/src/entrypoint.sh" ]
CMD "--help"

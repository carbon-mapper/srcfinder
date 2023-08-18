FROM mambaorg/micromamba:focal-cuda-11.8.0
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml && \
    micromamba clean --all --yes
COPY --chown=$MAMBA_USER:$MAMBA_USER . /mnt/cnn
WORKDIR /mnt/cnn

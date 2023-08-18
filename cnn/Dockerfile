FROM mambaorg/micromamba:jammy-cuda-12.2.0
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml && \
    micromamba clean --all --yes
WORKDIR /mnt/cnn
COPY --chown=$MAMBA_USER:$MAMBA_USER . /mnt/cnn
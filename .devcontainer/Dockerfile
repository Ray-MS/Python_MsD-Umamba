FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

ENV TZ=Etc/UTC \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    PIP_DEFAULT_TIMEOUT=1000 \
    PIP_RETRIES=10
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspaces

COPY requirements.txt /tmp/requirements.txt
ENV CAUSAL_CONV1D_FORCE_BUILD=TRUE \
    MAMBA_FORCE_BUILD=TRUE
RUN pip install \
    --no-build-isolation \
    --no-cache-dir \
    -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

CMD ["bash"]

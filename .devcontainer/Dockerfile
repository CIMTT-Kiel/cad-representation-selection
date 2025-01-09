FROM nvcr.io/nvidia/dgl:24.07-py3
WORKDIR /workspaces/tree-lstm 

RUN apt-get update && apt-get install -y \
    graphviz \
    graphviz-dev

COPY pyproject.toml .
COPY src ./src
RUN pip install -e .

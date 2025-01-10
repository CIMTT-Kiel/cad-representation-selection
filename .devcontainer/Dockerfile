# Base image from nvidia contains compatable versions of cuda and dgl
FROM nvcr.io/nvidia/dgl:24.07-py3  
    WORKDIR /workspaces/tree-lstm 

# Install packages for visualization of graphs
RUN apt-get update && apt-get install -y \
    graphviz \
    graphviz-dev

COPY pyproject.toml .
COPY src ./src
RUN pip install -e .

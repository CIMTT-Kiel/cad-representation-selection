# Base image from nvidia contains compatable versions of cuda and dgl
FROM nvcr.io/nvidia/dgl:24.07-py3  
WORKDIR /clear-shape 

# Install packages for visualization of graphs
RUN apt-get update && apt-get install -y \
     graphviz \
     graphviz-dev \
     libgl1

# # # Download and extract gmsh SDK
# RUN wget https://gmsh.info/bin/Linux/gmsh-4.12.2-Linux64-sdk.tgz \
#  && tar -xzf gmsh-4.12.2-Linux64-sdk.tgz -C /opt/ \
#  && rm gmsh-4.12.2-Linux64-sdk.tgz \
#  && ln -s /opt/gmsh-4.12.2-Linux64-sdk /opt/gmsh

# # # Add gmsh to PATH and PYTHONPATH
# ENV PATH="/opt/gmsh/bin:$PATH"
# ENV PYTHONPATH="/opt/gmsh/lib:$PYTHONPATH"

COPY . .

RUN pip install -e .

# == Add git completion ==
# Download the git completion script and place it in the home directory
RUN curl https://raw.githubusercontent.com/git/git/master/contrib/completion/git-completion.bash -o ~/.git-completion.bash
# register the git completion script in the bashrc file
RUN echo "if [ -f ~/.git-completion.bash ]; then . ~/.git-completion.bash; fi" >> ~/.bashrc
FROM continuumio/miniconda3

ENV CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes  

WORKDIR /app

RUN conda install -y -c conda-forge \
      python=3.12 \
      freecad \
      python-gmsh

RUN pip install --retries 10 --timeout 60 \
      open3d cascadio uvicorn fastapi trimesh


COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

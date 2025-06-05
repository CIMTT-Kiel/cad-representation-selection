FROM continuumio/miniconda3

WORKDIR /app

# Python 3.12 in der base-Umgebung installieren
RUN conda install -y python=3.12

RUN pip install open3d
RUN pip install cascadio
RUN conda install conda-forge::freecad
RUN pip install uvicorn
RUN pip install fastapi
RUN conda install conda-forge::python-gmsh
RUN pip install trimesh 


COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

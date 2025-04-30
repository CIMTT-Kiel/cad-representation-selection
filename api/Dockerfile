FROM continuumio/miniconda3

WORKDIR /app

RUN conda install conda-forge::freecad
RUN pip install uvicorn
RUN pip install fastapi
RUN conda install conda-forge::python-gmsh


COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

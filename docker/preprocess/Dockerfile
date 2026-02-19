FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

COPY requirements-preprocess.txt /workspace/requirements-preprocess.txt
RUN pip install --upgrade pip && pip install -r /workspace/requirements-preprocess.txt

COPY scripts/build_dataset.py /workspace/scripts/build_dataset.py

ENTRYPOINT ["python", "/workspace/scripts/build_dataset.py"]

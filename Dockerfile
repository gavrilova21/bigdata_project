FROM pytorch/pytorch:latest
WORKDIR /
RUN apt-get update \
     && apt-get install -y \
        libgl1-mesa-glx \
        libx11-xcb1 \
     && apt-get clean all \
     && rm -r /var/lib/apt/lists/*
RUN pip install mlflow==1.20.2
RUN pip install scikit-learn
RUN pip install numpy
RUN pip install scipy
RUN pip install pandas
RUN pip install pysqlite3
RUN pip install transformers
RUN pip --no-cache-dir install torch
RUN pip install Pillow
RUN pip install python-multipart
RUN pip install torchvision==0.12.0
RUN pip install tqdm
RUN pip install flask==2.1.2
RUN pip install tensorboard==2.9.0
RUN pip install onnx==1.11.0
RUN pip install onnxruntime==1.11.1


ENV BACKEND_URI sqlite:////mlflow/mlflow.db
ENV ARTIFACT_ROOT /mlflow/artifacts
ENV MLFLOW_TRACKING_URI=http://localhost:5000

CMD mlflow server --backend-store-uri ${BACKEND_URI} --default-artifact-root ${ARTIFACT_ROOT} --host 0.0.0.0 --port 5000

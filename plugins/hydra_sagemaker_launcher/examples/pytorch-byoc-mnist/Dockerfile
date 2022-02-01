# Build: sudo docker build -t <project_name> .
# Run: sudo docker run -v $(pwd):/workspace/project --gpus all -it --rm <project_name>

FROM nvidia/cuda:11.4.1-base-ubuntu18.04

ENV PYTHON_VERSION=3.8.11

ENV PATH /opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/conda/lib"

ENV PYTHONIOENCODING=UTF-8
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_AUTO_UPDATE_CONDA=false

ENV SAGEMAKER_PROGRAM ./_remote_invoke.py

RUN apt update
RUN apt install -y bash \
    build-essential \
    git \
    curl \
    ca-certificates \
    wget \
    && rm -rf /var/lib/apt/lists

# Install Miniconda and create main env
ADD https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh miniconda3.sh
RUN /bin/bash miniconda3.sh -b -p /opt/conda \
    && rm miniconda3.sh \
    && /opt/conda/bin/conda install -y -c anaconda \
    python=$PYTHON_VERSION \
    && /opt/conda/bin/conda clean -ya

RUN /opt/conda/bin/conda config --set ssl_verify False \
    && pip install --upgrade pip --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    && ln -s /opt/conda/bin/pip /usr/local/bin/pip3

# Install SageMaker dependencies for training inside containers
RUN pip install \
    --no-cache-dir sagemaker-pytorch-training

# Install requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && rm requirements.txt

CMD ["/bin/bash"]

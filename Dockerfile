FROM python:3.8-slim-buster

RUN apt-get update -y

# gcc compiler and opencv prerequisites
RUN apt-get -y install nano git build-essential libglib2.0-0 libsm6 libxext6 libxrender-dev

# Detectron2 prerequisites
RUN pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install cython
RUN pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# Detectron2 - CPU copy
RUN python -m pip install detectron2==0.1.1 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/index.html

# Development packages
RUN pip install flask flask-cors requests opencv-python

#ADD volume /
ADD vovnet /vovnet/
ADD centermask /centermask/
ADD train_net.py /

#ENTRYPOINT ["python", "train_net.py"]
# python train_net.py --eval-only MODEL.WEIGHTS /volume/configs/centermask2-V-39-eSE-FPN-ms-3x.pth
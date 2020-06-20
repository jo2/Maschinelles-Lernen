FROM nvidia/cuda:10.0-base-ubuntu18.04

RUN apt-get update -y

# gcc compiler and opencv prerequisites
RUN apt-get -y install nano git build-essential libglib2.0-0 libsm6 libxext6 libxrender-dev python3-pip

# Detectron2 prerequisites
RUN pip3 install torch==1.4.0+cu100 torchvision==0.5.0+cu100 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install cython
RUN pip3 install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

RUN pip3 --version
RUN python3 --version
# Detectron2 - CPU copy
RUN python3 -m pip install detectron2==0.1.1+cu100 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu100/torch1.4/index.html

# Development packages
RUN pip3 install flask flask-cors requests opencv-python

#ADD volume /volume
ADD vovnet /vovnet/
ADD centermask /centermask/
ADD train_net.py /

#CMD ["python", "train_net.py"]
# python train_net.py --eval-only MODEL.WEIGHTS /volume/configs/centermask2-V-39-eSE-FPN-ms-3x.pth
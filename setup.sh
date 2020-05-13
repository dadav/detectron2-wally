#!/bin/bash

if [[ "$#" -ne 1 ]]; then
  echo "please give me 1 argument (cpu or gpu)"
  exit 1
fi

# create a venv
[[ -z "$VIRTUAL_ENV" ]] && [[ -z "$SKIP_VIRTUAL_ENV" ]] && python3 -m venv venv && source venv/bin/activate

# pip
if [[ "$1" == "cpu" ]]; then
  pip install -U detectron2==0.1.2+cpu -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/index.html
  pip install -U torch==1.5+cpu torchvision==0.6+cpu -f https://download.pytorch.org/whl/torch_stable.html
else
  pip install -U torch==1.5 torchvision==0.6 -f https://download.pytorch.org/whl/cu101/torch_stable.html
  pip install detectron2==0.1.2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/index.html
fi
pip install cython pyyaml==5.1
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install opencv-python
pip install streamlit

# split dataset
#ls images/*.jpg | shuf -n 10 | xargs -r -I{} mv {} images/val/
ls images/*.jpg | xargs -r -I{} mv {} images/train/

# recover moved files
git checkout -- images/*.jpg

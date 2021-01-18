# Semantic segmentation reference training scripts
Tested on `PyTorch:1.7.0`.

## docker
* Python: 3.8
* PyTorch: 1.7.0
* `http://ip:9000/?token=hi` for `dev`
* `/usr/sbin/sshd -D -p 9000` for `ssh` mode
* `python /workspace/app_tornado.py 9000 ${@:2}` for `app` mode

```
docker pull flystarhe/python:3.8-torch1.7.0
docker run --gpus all -d -p 9000:9000 --ipc=host --name test -v "$(pwd)":/workspace flystarhe/python:3.8-torch1.7.0
```

## fcn_resnet50
```
import os
PYSEG_HOME = "/workspace/pyseg-pytorch"
os.environ["PYSEG_HOME"] = PYSEG_HOME
!cd {PYSEG_HOME} && git log -1 --oneline
os.environ["MKL_THREADING_LAYER"] = "GNU"

import time
EXPERIMENT_NAME = time.strftime("T%m%d_%H%M")
EXPERIMENT_NAME
```

Train:
```
DATA_ROOT = "/workspace/data/data_xxxx_coco"

ARG_DIST = "-m torch.distributed.launch --nproc_per_node=2"

ARG_TRAIN = "--data-path {} --output-dir results/{}".format(DATA_ROOT, EXPERIMENT_NAME)
ARG_TRAIN += (" --model fcn_resnet50"
              " --aux-loss"
              " -b 4"
              " -j 8"
              " --epochs 30"
              " --lr 0.005"
              " --pretrained")

!PYTHONPATH={PYSEG_HOME} python {ARG_DIST} --use_env tools/toy/py_train.py {ARG_TRAIN}
```

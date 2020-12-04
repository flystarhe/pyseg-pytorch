# Semantic segmentation reference training scripts
Tested on `PyTorch:1.7.0`.

## docker
```
docker pull flystarhe/pytorch:1.7.0
docker run --gpus all -d -p 9000:9000 -p 9001:9001 --ipc=host --name test -v "$(pwd)":/workspace flystarhe/pytorch:1.7.0
```

## fcn_resnet50
```
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

PYSEG_HOME = "/workspace"
os.environ["PYSEG_HOME"] = PYSEG_HOME
os.chdir(PYSEG_HOME)
!git log -1 --oneline

import time
EXPERIMENT_NAME = time.strftime("T%m%d_%H%M")
EXPERIMENT_NAME
```

Train:
```
ARGS = "--data-path data/VOC2007_ --output-dir results/{}".format(EXPERIMENT_NAME)

ARGS += (" --model fcn_resnet50"
         " --aux-loss"
         " -b 4"
         " -j 8"
         " --epochs 30"
         " --lr 0.005"
         " --pretrained")

!PYTHONPATH="$(pwd)":{PYSEG_HOME} python -m torch.distributed.launch --nproc_per_node=2 --use_env tools/toy/py_train.py {ARGS}
```

# Semantic segmentation reference training scripts

```
%matplotlib inline
import os

PYSEG_HOME = "/data/tmp/gits/pyseg-pytorch"
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["PYSEG_HOME"] = PYSEG_HOME
os.chdir(PYSEG_HOME)
!pwd

import time
EXPERIMENT_NAME = time.strftime("T%m%d_%H%M")
EXPERIMENT_NAME
```

## fcn_resnet50
```
ARGS = "--data-path /data/tmp/gits/data/VOC2007_ --output-dir tmp/{}".format(EXPERIMENT_NAME)

ARGS += (" --model fcn_resnet50"
         " --aux-loss"
         " -b 4"
         " -j 8"
         " --epochs 30"
         " --lr 0.005"
         " --pretrained")

!PYTHONPATH=`pwd`:{PYSEG_HOME} python -m torch.distributed.launch --nproc_per_node=2 --use_env tools/toy/py_train.py {ARGS}
```

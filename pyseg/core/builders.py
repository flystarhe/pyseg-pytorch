import torch
from pyseg.core.config import cfg
from pyseg.models.toy import Toy


# Supported models
_models = {"toy": Toy}


# Supported loss functions
_loss_funs = {
    "l1": torch.nn.L1Loss, "mse": torch.nn.MSELoss, "cross_entropy": torch.nn.CrossEntropyLoss,
    "nll": torch.nn.NLLLoss, "bce": torch.nn.BCELoss, "bce_logits": torch.nn.BCEWithLogitsLoss,
}


def get_model():
    """Gets the model class specified in the config."""
    err_str = "Model type '{}' not supported"
    assert cfg.MODEL.TYPE in _models.keys(), err_str.format(cfg.MODEL.TYPE)
    return _models[cfg.MODEL.TYPE]


def get_loss_fun():
    """Gets the loss function class specified in the config."""
    err_str = "Loss function type '{}' not supported"
    assert cfg.MODEL.LOSS_FUN in _loss_funs.keys(), err_str.format(cfg.MODEL.LOSS_FUN)
    return _loss_funs[cfg.MODEL.LOSS_FUN]


def build_model():
    """Builds the model."""
    return get_model()()


def build_loss_fun():
    """Build the loss function."""
    return get_loss_fun()()

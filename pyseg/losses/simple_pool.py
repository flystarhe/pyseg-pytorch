import numpy as np
import torch


def test():
    x = torch.randn(3, 4, requires_grad=True)
    y = torch.zeros_like(x, requires_grad=False)

    r1 = torch.where(x > 0, x, y)

    mask = x.ge(0.5)
    r2 = torch.masked_select(x, mask)

    indices = torch.tensor([0, 2])
    r3 = torch.index_select(x, 0, indices)
    return r1, r2, r3


def split(a, b, parts=3, stride=8):
    a, b = a / stride, b / stride
    ps = np.linspace(a, b, parts + 1).round().astype("int32")
    return [(ps[i], ps[i + 1]) for i in range(parts) if ps[i] < ps[i + 1]]


def make_rois(feat, x1, y1, x2, y2, parts, stride):
    x_pairs = split(x1, x2, parts, stride)
    y_pairs = split(y1, y2, parts, stride)

    rois = []
    scores = []
    for _x1, _x2 in x_pairs:
        for _y1, _y2 in y_pairs:
            rois.append([_x1, _y1, _x2, _y2])
            scores.append(torch.max(feat[_y1:_y2, _x1:_x2]).item())
    return rois, scores


def get_rois(feat, boxes, parts, stride):
    """
    Note:
        box = [x1, y1, x1 + w, y1 + h]
    """
    rois = []
    for box in boxes:
        _rois, _score = make_rois(feat, *box, parts, stride)
        for ind in np.argsort(_score)[::-1][:parts]:
            rois.append(_rois[ind])
    return rois


def get_fg_mask(feat, boxes, stride):
    """
    Note:
        box = [x1, y1, x1 + w, y1 + h]
    """
    mask = torch.zeros_like(feat, dtype=torch.int32)
    for x1, y1, x2, y2 in boxes:
        x1 = int(np.ceil(x1 / stride))
        y1 = int(np.ceil(y1 / stride))
        x2 = int(np.floor(x2 / stride))
        y2 = int(np.floor(y2 / stride))
        for yi in range(y1, y2):
            shift = torch.argmax(feat[yi, x1:x2])
            mask[..., yi, x1 + shift] = 1
    return mask.bool()


def get_bg_mask(feat, boxes, stride):
    """
    Note:
        box = [x1, y1, x1 + w, y1 + h]
    """
    mask = torch.ones_like(feat, dtype=torch.int32)
    for x1, y1, x2, y2 in boxes:
        x1 = max(0, int(x1 / stride) - 1)
        y1 = max(0, int(y1 / stride) - 1)
        x2 = int(np.ceil(x2 / stride) + 1)
        y2 = int(np.ceil(y2 / stride) + 1)
        mask[..., y1:y2, x1:x2] = 0
    return mask.bool()


def get_mask(feat, boxes, stride):
    """
    Note:
        box = [x1, y1, x1 + w, y1 + h]
    """
    bg_mask = torch.ones_like(feat, dtype=torch.int32)
    fg_mask = torch.zeros_like(feat, dtype=torch.int32)

    for x1, y1, x2, y2 in boxes:
        x1 = int(np.ceil(x1 / stride))
        y1 = int(np.ceil(y1 / stride))
        x2 = int(np.floor(x2 / stride))
        y2 = int(np.floor(y2 / stride))

        for yi in range(y1, y2):
            shift = torch.argmax(feat[yi, x1:x2])
            fg_mask[..., yi, x1 + shift] = 1

        x1, y1 = max(0, x1 - 1), max(0, y1 - 1)
        bg_mask[..., y1:y2 + 1, x1:x2 + 1] = 0

    return bg_mask.bool(), fg_mask.bool()


def get_points(mask, topk):
    inds = torch.nonzero(mask.view(-1), as_tuple=True)[0]
    return inds[torch.randperm(inds.numel())[:topk]]


def _splits(a, b, parts):
    ps = np.linspace(a, b, parts + 1).round().astype("int32")
    return [(ps[i], ps[i + 1]) for i in range(parts) if ps[i] < ps[i + 1]]


def _points(topk, feat, x1, y1, x2, y2):
    _y_pairs = _splits(y1, y2, topk)
    _x_pairs = _splits(x1, x2, topk)

    points, scores = [], []
    for _y1, _y2 in _y_pairs:
        for _x1, _x2 in _x_pairs:
            _shift = torch.argmax(feat[_y1:_y2, _x1:_x2]).item()
            _shift_y, _shift_x = divmod(_shift, _x2 - _x1)
            cy, cx = _y1 + _shift_y, _x1 + _shift_x
            scores.append(feat[cy, cx].item())
            points.append((cy, cx))

    topk = max(len(_y_pairs), len(_x_pairs))
    return [points[i] for i in np.argsort(scores)[::-1][:topk]]


def create_mask(topk, feat, boxes, stride):
    """
    Note:
        box = [x1, y1, x1 + w, y1 + h]
    """
    bg_mask = torch.ones_like(feat, dtype=torch.int32)
    fg_mask = torch.zeros_like(feat, dtype=torch.int32)

    for x1, y1, x2, y2 in boxes:
        x1 = int(np.floor(x1 / stride))
        y1 = int(np.floor(y1 / stride))
        x2 = int(np.ceil(x2 / stride))
        y2 = int(np.ceil(y2 / stride))

        for cy, cx in _points(topk, feat, x1, y1, x2, y2):
            fg_mask[..., cy, cx] = 1

        x1, y1 = max(0, x1 - 1), max(0, y1 - 1)
        bg_mask[..., y1:y2 + 1, x1:x2 + 1] = 0

    return bg_mask.bool(), fg_mask.bool()

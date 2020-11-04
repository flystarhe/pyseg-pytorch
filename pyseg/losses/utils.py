import torch
import numpy as np
import torch.nn.functional as F


def _splits(a, b, parts):
    ps = torch.linspace(a, b, parts + 1, dtype=torch.int).tolist()
    return [(ps[i], ps[i + 1]) for i in range(parts) if ps[i] < ps[i + 1]]


def _points(feat, topk, x1, y1, x2, y2):
    # feat (Tensor[H, W]): type
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


def _balance_target(target, weight):
    # target, weight (Tensor[H, W]): type
    _, w = target.size()
    negative_mask = target.eq(0)
    n_positive = target.gt(0).sum().item()

    n_top_max = max(w, n_positive * 2)
    if negative_mask.sum().item() > n_top_max:
        probs, _ = weight[negative_mask].sort(descending=True)
        target[negative_mask * weight.lt(probs[n_top_max])] = -100

    return target


def _make_target(s, topk, feats, boxes, labels=None, balance=False):
    # feats (Tensor[K, H, W])： `0` means `_BG`, the C categories in `[1, K-1]`
    # labels (List[int]): where each value is `1 <= labels[i] <= C`
    # notes: assume `cross_entropy(ignore_index=-100)`
    if labels is None:
        labels = [1 for _ in boxes]

    _, h, w = feats.size()
    feats = F.softmax(feats, dim=0)
    masks = torch.zeros_like(feats, dtype=torch.uint8)
    for (x1, y1, x2, y2), i in zip(boxes, labels):
        x1 = int(np.floor(x1 * s))
        y1 = int(np.floor(y1 * s))
        x2 = int(np.ceil(x2 * s))
        y2 = int(np.ceil(y2 * s))

        masks[i, y1:y2, x1:x2] = 2
        x2, y2 = min(w, x2), min(h, y2)
        for cy, cx in _points(feats[i], topk, x1, y1, x2, y2):
            masks[i, cy, cx] = 1

    target = masks.argmax(0)
    target[masks.sum(0) < 1] = 0
    target[masks.sum(0) > 1] = -100

    if balance:
        target = _balance_target(target, feats[0])

    return target

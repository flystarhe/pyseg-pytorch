import torch
import numpy as np


def _splits(a, b, parts):
    ps = torch.linspace(a, b, parts + 1, dtype=torch.int).tolist()
    return [(ps[i], ps[i + 1]) for i in range(parts) if ps[i] < ps[i + 1]]


def _points(feat, topk, x1, y1, x2, y2):
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


def _make_target(topk, feats, boxes, labels, factor):
    # feats (Tensor[*, H, W])： 0 is bg, [1, C] with C classes
    # labels (List[int]): where each value is `1 <= labels[i] <= C`
    inds = labels
    if feats.size(0) == 1:
        inds = [0 for _ in labels]

    target = torch.zeros_like(feats[0], dtype=torch.long)

    feats = feats.softmax(dim=0)
    for (x1, y1, x2, y2), ind in zip(boxes, inds):
        x1 = int(np.floor(x1 * factor))
        y1 = int(np.floor(y1 * factor))
        x2 = int(np.ceil(x2 * factor))
        y2 = int(np.ceil(y2 * factor))

        x0, y0 = max(0, x1 - 1), max(0, y1 - 1)
        target[y0:y2 + 1, x0:x2 + 1] = -1

        for cy, cx in _points(feats[ind], topk, x1, y1, x2, y2):
            target[cy, cx] = ind

    return target


def _balance_target(targets):
    # List[Tensor[H, W]] or Tensor[N, H, W]
    if isinstance(targets, list):
        targets = torch.stack(targets, 0)

    negative_mask = targets.eq(0)
    n_positive = targets.gt(0).sum()
    if negative_mask.sum() > n_positive:
        probs = targets[negative_mask].sort(descending=True)[0]
        targets[negative_mask * targets.lt(probs[n_positive])] = -1

    return targets

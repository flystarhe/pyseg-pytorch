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
    # target, weight (Tensor[H, W] or Tensor[C, H, W]): type
    w = target.size()[-1]
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
    feats = F.softmax(feats, dim=0)

    if labels is None:
        indices = [1 for _ in boxes]
    else:
        indices = [label for label in labels]

    _, h, w = feats.size()
    masks = torch.zeros_like(feats, dtype=torch.uint8)
    for (x1, y1, x2, y2), i in zip(boxes, indices):
        x1 = int(np.floor(x1 * s))
        y1 = int(np.floor(y1 * s))
        x2 = int(np.ceil(x2 * s))
        y2 = int(np.ceil(y2 * s))

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        masks[i, y1:y2, x1:x2] = 2
        for cy, cx in _points(feats[i], topk, x1, y1, x2, y2):
            masks[i, cy, cx] = 1

    target = masks.argmax(0)

    mask = masks.sum(0)
    target[mask == 0] = 0
    target[mask > 1] = -100

    if balance:
        target = _balance_target(target, 1.0 - feats[0])

    return target


def cross_entropy_target(s, topk, feats, boxes, labels=None, balance=False):
    # feats (Tensor[K, H, W])： `0` means `_BG`, the C categories in `[1, K-1]`
    # labels (List[int]): where each value is `1 <= labels[i] <= C`
    # notes: assume `cross_entropy(ignore_index=-100)`
    # return (Tensor[H, W]): type
    feats = F.softmax(feats, dim=0)

    if labels is None:
        indices = [1 for _ in boxes]
    else:
        indices = [label for label in labels]
    boxes = sorted(boxes, key=lambda x: (x[3] - x[1]) * (x[2] - x[0]), reverse=True)

    _, h, w = feats.size()
    masks = torch.zeros_like(feats, dtype=torch.uint8)
    for (x1, y1, x2, y2), i in zip(boxes, indices):
        x1 = int(np.floor(x1 * s))
        y1 = int(np.floor(y1 * s))
        x2 = int(np.ceil(x2 * s))
        y2 = int(np.ceil(y2 * s))

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        masks[i, y1:y2, x1:x2] = 2
        for cy, cx in _points(feats[i], topk, x1, y1, x2, y2):
            masks[i, cy, cx] = 1

    target = masks.argmax(0)

    mask = masks.sum(0)
    target[mask == 0] = 0
    target[mask > 1] = -100

    if balance:
        target = _balance_target(target, 1.0 - feats[0])

    return target


def binary_cross_entropy_target(s, topk, feats, boxes, labels=None, balance=False):
    # feats (Tensor[C, H, W])：no channel `_BG`, the C categories in `[0, C-1]`
    # labels (List[int]): where each value is `1 <= labels[i] <= C`
    # notes: assume `cross_entropy(ignore_index=-100)`
    # return (Tensor[C, H, W]): type
    if labels is None:
        indices = [1 for _ in boxes]
    else:
        indices = [label for label in labels]
    boxes = sorted(boxes, key=lambda x: (x[3] - x[1]) * (x[2] - x[0]), reverse=True)

    _, h, w = feats.size()
    masks = torch.zeros_like(feats, dtype=torch.uint8)
    for (x1, y1, x2, y2), i in zip(boxes, indices):
        x1 = int(np.floor(x1 * s))
        y1 = int(np.floor(y1 * s))
        x2 = int(np.ceil(x2 * s))
        y2 = int(np.ceil(y2 * s))

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        masks[i, y1:y2, x1:x2] = 2
        for cy, cx in _points(feats[i], topk, x1, y1, x2, y2):
            masks[i, cy, cx] = 1

    target = masks.clone()

    target[masks > 1] = -100

    if balance:
        target = _balance_target(target, feats)

    return target

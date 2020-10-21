import datetime
import numpy as np
import os
import time
import torch
import torch.utils.data

import apis
import utils
from tools.toy import apis
from tools.toy import utils


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


def _create_mask(topk, feat, boxes, s):
    """
    Note:
        box = [x1, y1, x1 + w, y1 + h]
    """
    bg_mask = torch.ones_like(feat, dtype=torch.int32)
    fg_mask = torch.zeros_like(feat, dtype=torch.int32)

    for x1, y1, x2, y2 in boxes:
        x1 = int(np.floor(x1 * s))
        y1 = int(np.floor(y1 * s))
        x2 = int(np.ceil(x2 * s))
        y2 = int(np.ceil(y2 * s))

        for cy, cx in _points(topk, feat, x1, y1, x2, y2):
            fg_mask[..., cy, cx] = 1

        x1, y1 = max(0, x1 - 1), max(0, y1 - 1)
        bg_mask[..., y1:y2 + 1, x1:x2 + 1] = 0

    return bg_mask.bool(), fg_mask.bool()


def _criterion_single(output, target, s):
    bg_mask, fg_mask = _create_mask(3, output.detach(), target["bboxes"], s)

    loss_bg = 0.
    if bg_mask.any():
        x = torch.masked_select(output, bg_mask)
        y = torch.zeros_like(x, requires_grad=False)
        loss_bg = F.binary_cross_entropy_with_logits(x, y)

    loss_fg = 0.
    if fg_mask.any():
        x = torch.masked_select(output, fg_mask)
        y = torch.ones_like(x, requires_grad=False)
        loss_fg = F.binary_cross_entropy_with_logits(x, y)

    return (loss_bg + loss_fg) * 0.5


def _criterion(outputs, targets, shape):
    n, _, _, w = outputs.size()

    loss = 0.
    f = 1.0 / n
    s = w / shape[-1]
    for img_id in range(n):
        loss = loss + f * _criterion_single(outputs[img_id], targets[img_id], s)
    return loss


def criterion(outputs, targets):
    input_shape = outputs["input_shape"]
    loss = _criterion(outputs["out"], targets, input_shape)
    if "aux" in outputs:
        return loss + 0.5 * _criterion(outputs["aux"], targets, input_shape)
    return loss


def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Eval:"
    with torch.no_grad():
        for images, targets in metric_logger.log_every(data_loader, 100, header):
            outputs = model(images.to(device))
            outputs = outputs["out"]

            confmat.update(targets.flatten(), outputs.argmax(1).flatten())

        confmat.reduce_from_all_processes()

    return confmat


def test_one_epoch(model, data_loader, device, output_dir):
    # draw box and save image, count boxes `box.max < thr`
    return output_dir

    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    with torch.no_grad():
        for images, targets in metric_logger.log_every(data_loader, 100, header):
            outputs = model(images.to(device))
            outputs = outputs["out"]


def train_one_epoch(model, optimizer, data_loader, lr_scheduler, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    header = "Epoch: [{}]".format(epoch)
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        outputs = model(images.to(device))
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_scheduler.step()

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])


def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    dataset = apis.ToyDataset(args.data_path, "train")
    dataset_test = apis.ToyDataset(args.data_path, "val")

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers,
        collate_fn=apis.collate_fn, drop_last=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=apis.collate_fn, drop_last=False)

    model = apis.get_model(backbone_name="resnet50", num_classes=1, aux_loss=args.aux_loss,
                           pretrained=args.pretrained, classes=dataset.classes)

    model.to(device)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.test_only:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        test_one_epoch(model, data_loader_test, device=device, output_dir=args.output_dir)
        return

    params_to_optimize = [
        {"params": [p for p in model_without_ddp.backbone.parameters() if p.requires_grad]},
        {"params": [p for p in model_without_ddp.classifier.parameters() if p.requires_grad]},
    ]
    if args.aux_loss:
        params = [p for p in model_without_ddp.aux_classifier.parameters() if p.requires_grad]
        params_to_optimize.append({"params": params, "lr": args.lr * 10})
    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, data_loader, lr_scheduler, device, epoch, args.print_freq)
        utils.save_on_master(
            {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args
            },
            os.path.join(args.output_dir, "model_{}.pth".format(epoch)))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))

    test_one_epoch(model, data_loader_test, device=device, output_dir=args.output_dir)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Total time {}".format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="PyTorch Segmentation Training")

    parser.add_argument("--data-path", help="dataset path")
    parser.add_argument("--device", default="cuda", help="device")
    parser.add_argument("--model", default="fcn_resnet50", help="model")
    parser.add_argument("--aux-loss", action="store_true", help="auxiliar loss")
    parser.add_argument("-b", "--batch-size", default=8, type=int)
    parser.add_argument("--epochs", default=30, type=int)

    parser.add_argument("-j", "--workers", default=16, type=int)
    parser.add_argument("--lr", default=0.01, type=float, help="lr")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--wd", "--weight-decay", default=1e-4, type=float, dest="weight_decay")
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--output-dir", default=".", help="path where to save")
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--test-only", action="store_true")

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)

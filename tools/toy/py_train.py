import datetime
import os
import time
import torch
import torch.nn as nn
import torch.utils.data


import apis
import utils
from tools.toy import apis
from tools.toy import utils


def _transform(output, target, shape):
    # output (Tensor[N, C, H, W]): type
    n, _, _, w = output.size()
    s = w / shape[-1]
    topk = 3

    output = output.detach()

    _target = []
    for i in range(n):
        _target.append(apis._make_target(s,
                                         topk,
                                         output[i],
                                         target[i]["bboxes"],
                                         None,
                                         True))

    return torch.stack(_target, 0).to(output.device)


def criterion(output, target):
    _shape = output["input_shape"]

    _output = output["out"]
    _target = _transform(_output, target, _shape)
    loss = nn.functional.cross_entropy(_output, _target, ignore_index=-100)

    if "aux" in output:
        _output = output["aux"]
        _target = _transform(_output, target, _shape)
        return loss + 0.5 * nn.functional.cross_entropy(_output, _target, ignore_index=-100)

    return loss


def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            output = model(image.to(device))
            _shape = output["input_shape"]

            _output = output["out"]
            _target = _transform(_output, target, _shape)
            confmat.update(_target.flatten(), _output.argmax(1).flatten())

        confmat.reduce_from_all_processes()

    return confmat


def test_one_epoch(model, data_loader, device, output_dir):
    # draw box and save image, count boxes `box.max < thr`
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            output = model(image.to(device))
            vals, inds = output["out"].max(dim=1)


def train_one_epoch(model, optimizer, data_loader, lr_scheduler, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    header = "Epoch: [{}]".format(epoch)
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        output = model(image.to(device))
        loss = criterion(output, target)

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

    model = apis.get_model(backbone_name="resnet50", aux_loss=args.aux_loss, pretrained=args.pretrained)
    num_classes = len(model.classes)

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
        confmat = evaluate(model, data_loader_test, device=device, num_classes=num_classes)
        print(confmat)
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


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="PyTorch Segmentation Training")

    parser.add_argument("--data-path", help="dataset path")
    parser.add_argument("--device", default="cuda", help="device")
    parser.add_argument("--model", default="fcn_resnet50", help="model")
    parser.add_argument("--aux-loss", action="store_true", help="auxiliar loss")
    parser.add_argument("-b", "--batch-size", default=8, type=int)
    parser.add_argument("-j", "--workers", default=16, type=int)
    parser.add_argument("--epochs", default=30, type=int)

    parser.add_argument("--lr", default=0.01, type=float, help="initial lr")
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

from argparse import ArgumentParser
from typing import List
import time
import numpy as np
from tqdm import tqdm
import os
import json
import torch as ch
from torch.optim import lr_scheduler
import torchvision


import projunn

from fastargs import get_current_config, Param, Section
from fastargs.decorators import param
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import (
    RandomHorizontalFlip,
    Cutout,
    RandomTranslate,
    Convert,
    ToDevice,
    ToTensor,
    ToTorchImage,
)
from ffcv.transforms.common import Squeeze


Section("training", "Hyperparameters").params(
    lr=Param(float, "The learning rate to use", required=True),
    epochs=Param(int, "Number of epochs to run for", required=True),
    lr_peak_epoch=Param(int, "Peak epoch for cyclic lr", required=True),
    batch_size=Param(int, "Batch size", default=512),
    momentum=Param(float, "Momentum for SGD", default=0.9),
    weight_decay=Param(float, "l2 weight decay", default=5e-4),
    label_smoothing=Param(float, "Value of label smoothing", default=0.1),
    num_workers=Param(int, "The number of workers", default=8),
    scheduler=Param(str, "The scheduler", default="triangle"),
    lr_tta=Param(
        bool,
        "Test time augmentation by averaging with horizontally flipped version",
        default=True,
    ),
)


Section("model", "Hyperparameters").params(
    name=Param(str, "the name of the model", default="resnet9"),
)

Section("data", "data related stuff").params(
    classes=Param(int, "number of classes", required=True),
    log_dir=Param(str, "where to save results", required=True),
    train_dataset=Param(str, ".dat file to use for training", required=True),
    val_dataset=Param(str, ".dat file to use for validation", required=True),
)


@param("data.train_dataset")
@param("data.val_dataset")
@param("training.batch_size")
@param("training.num_workers")
def make_dataloaders(
    train_dataset=None,
    val_dataset=None,
    batch_size=None,
    num_workers=None,
):
    paths = {"train": train_dataset, "test": val_dataset}

    start_time = time.time()
    CIFAR_MEAN = [125.307, 122.961, 113.8575]
    CIFAR_STD = [51.5865, 50.847, 51.255]
    loaders = {}

    for name in ["train", "test"]:
        label_pipeline: List[Operation] = [
            IntDecoder(),
            ToTensor(),
            ToDevice("cuda:0"),
            Squeeze(),
        ]
        image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]
        if name == "train":
            image_pipeline.extend(
                [
                    RandomHorizontalFlip(),
                    RandomTranslate(padding=2),
                ]
            )
        image_pipeline.extend(
            [
                ToTensor(),
                ToDevice("cuda:0", non_blocking=True),
                ToTorchImage(),
                Convert(ch.float32),
                torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ]
        )

        ordering = OrderOption.RANDOM if name == "train" else OrderOption.SEQUENTIAL

        loaders[name] = Loader(
            paths[name],
            batch_size=batch_size,
            num_workers=num_workers,
            order=ordering,
            drop_last=(name == "train"),
            pipelines={"image": image_pipeline, "label": label_pipeline},
        )

    return loaders, start_time


@param("model.name")
def construct_model(name):
    if name == "resnet9":
        model = projunn.models.ResNet9(3, 10)
    model = model.cuda()
    return model


@param("training.lr")
@param("training.epochs")
@param("training.momentum")
@param("training.weight_decay")
@param("training.label_smoothing")
@param("training.lr_peak_epoch")
@param("training.scheduler")
def train(
    model,
    loaders,
    lr=None,
    epochs=None,
    label_smoothing=None,
    momentum=None,
    weight_decay=None,
    lr_peak_epoch=None,
    scheduler=None,
):
    def projector(param, update):
        a, b = projunn.utils.LSI_approximation(update, k=1)
        update = projunn.utils.projUNN_T(param.data, a, b)
        return update

    opt = projunn.optimizers.RMSprop(
        model.parameters(),
        projector=projector,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    iters_per_epoch = len(loaders["train"])
    # Cyclic LR with single triangle
    if scheduler == "triangle":
        lr_schedule = np.interp(
            np.arange((epochs + 1) * iters_per_epoch),
            [0, lr_peak_epoch * iters_per_epoch, epochs * iters_per_epoch],
            [0, 1, 0],
        )
        scheduler = lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)
    elif scheduler == "step":
        scheduler1 = lr_scheduler.LinearLR(
            opt,
            start_factor=0.001,
            end_factor=1,
            total_iters=epochs * iters_per_epoch,
        )
        scheduler2 = lr_scheduler.StepLR(opt, (epochs * iters_per_epoch) // 3, 0.2)
        scheduler = lr_scheduler.SequentialLR(
            opt,
            schedulers=[scheduler1, scheduler2],
            milestones=[lr_peak_epoch * iters_per_epoch],
        )

    loss_fn = ch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    for _ in range(epochs):
        for ims, labs in tqdm(loaders["train"]):
            opt.zero_grad(set_to_none=True)

            out = model(ims)
            loss = loss_fn(out, labs)
            loss.backward()
            opt.step()
            scheduler.step()


@param("training.lr_tta")
def evaluate(model, loaders, lr_tta=False):
    model.eval()
    results = {}
    with ch.no_grad():
        for name in ["train", "test"]:
            total_correct, total_num = 0.0, 0.0
            for ims, labs in tqdm(loaders[name]):
                out = model(ims)
                if lr_tta:
                    out += model(ch.fliplr(ims))
                total_correct += out.argmax(1).eq(labs).sum().cpu().item()
                total_num += ims.shape[0]
            results[name] = np.round(total_correct / total_num * 100, 4)
            print(f"{name} accuracy: {total_correct / total_num * 100:.1f}%")
    return results


if __name__ == "__main__":
    config = get_current_config()
    parser = ArgumentParser(description="Fast CIFAR training with projUNN constraint")
    config.augment_argparse(parser)
    # Also loads from args.config_path if provided
    config.collect_argparse_args(parser)
    config.validate(mode="stderr")
    config.summary()

    loaders, start_time = make_dataloaders()
    model = construct_model()
    train(model, loaders)
    print(f"Total time: {time.time() - start_time:.5f}")
    results = evaluate(model, loaders)
    results["lr"] = config["training.lr"]
    results["weight_decay"] = config["training.weight_decay"]
    results["epochs"] = config["training.epochs"]
    results["label_smoothing"] = config["training.label_smoothing"]
    results["momentum"] = config["training.momentum"]
    results["scheduler"] = config["training.scheduler"]
    filename = os.path.join(
        config["data.log_dir"],
        f"{config['data.classes']}_{config['model.name']}.txt",
    )
    print(filename, results)
    with open(filename, "a+") as fd:
        fd.write(json.dumps(results) + "\n")
        fd.flush()

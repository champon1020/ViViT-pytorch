import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Loss
from timesformer_pytorch import TimeSformer
from torch.utils.data import DataLoader

import wandb
from config import Config
from ucf101 import UCF101
from vivit import ViViT

wandb_online = True

cfg = Config()

"""
model = ViViT(
    dim=512,
    image_size=cfg.image_size,
    patch_size=16,
    num_classes=101,
    num_frames=cfg.n_frames,
    depth=12,
    heads=4,
    pool="cls",
    in_channels=3,
    dim_head=64,
    dropout=0.1,
).cuda()
"""
model = TimeSformer(
    dim=512,
    image_size=cfg.image_size,
    patch_size=16,
    num_frames=cfg.n_frames,
    num_classes=101,
    depth=12,
    heads=8,
    dim_head=64,
    attn_dropout=0.1,
    ff_dropout=0.1,
).cuda()

model = nn.DataParallel(model)
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=cfg.base_lr,
    momentum=0.9,
    nesterov=True,
    weight_decay=1e-3,
)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg.multistep_milestones)
ce_loss_fn = nn.CrossEntropyLoss()


def onehot_label(class_num: torch.Tensor):
    return F.one_hot(class_num, num_classes=101)


print(sum(p.numel() for p in model.parameters()))


def train_step(engine, batch):
    # return 0  # debug
    model.train()
    optimizer.zero_grad()
    video, class_num = batch["video"].cuda(), batch["class"].cuda()
    pred = model(video)
    pred = F.softmax(pred, dim=1)
    loss = ce_loss_fn(pred, class_num)
    # print(torch.argmax(pred, dim=1), class_num)
    loss.backward()
    optimizer.step()
    scheduler.step()
    # torch.cuda.empty_cache()
    return loss.item()


trainer = Engine(train_step)


def validation_step(engine, batch):
    # return torch.rand(16, 101), torch.zeros(16).long()  # debug
    model.eval()
    with torch.no_grad():
        video, class_num = batch["video"].cuda(), batch["class"].cuda()
        pred = model(video)
        pred = F.softmax(pred, dim=1)
        # torch.cuda.empty_cache()

    return pred, class_num


evaluator = Engine(validation_step)

accuracy_metric = Accuracy()
accuracy_metric.attach(evaluator, "accuracy")
ce_loss_metric = Loss(ce_loss_fn)
ce_loss_metric.attach(evaluator, "loss")


@trainer.on(Events.ITERATION_COMPLETED)
def log_training_loss(engine):
    e = engine.state.epoch
    i = engine.state.iteration
    loss = engine.state.output
    print(f"Epoch: {e} / {cfg.epochs} : {i} - Loss: {loss:.5f}")
    # if wandb_online:
    #   wandb.log({"loss": loss})


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(engine):
    state = evaluator.run(train_loader)
    metrics = state.metrics
    loss = metrics["loss"]
    accuracy = metrics["accuracy"]
    e = engine.state.epoch
    print(f"Training Results - Loss: {loss:.5f}, Avg accuracy: {accuracy:.5f}")
    if wandb_online:
        wandb.log({"train_loss": loss, "train_accuracy": accuracy})


@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(engine):
    state = evaluator.run(val_loader)
    metrics = state.metrics
    loss = metrics["loss"]
    accuracy = metrics["accuracy"]
    print(f"Valiation Results - Loss: {loss:.5f}, Avg accuracy: {accuracy:.5f}")
    if wandb_online:
        wandb.log({"validation_loss": loss, "validation_accuracy": accuracy})


if wandb_online:
    wandb.init(project="vivit", name=f"vivit-{datetime.datetime.now()}")


train_loader = DataLoader(
    UCF101(
        "./dataset/UCF101",
        "./dataset/ucfTrainTestlist/classInd.txt",
        [
            f"./dataset/ucfTrainTestlist/trainlist01.txt",
            # f"./dataset/ucfTrainTestlist/trainlist02.txt",
        ],
        cfg.n_frames,
        cfg.image_size,
    ),
    cfg.batch_size,
    shuffle=True,
)

val_loader = DataLoader(
    UCF101(
        "./dataset/UCF101",
        "./dataset/ucfTrainTestlist/classInd.txt",
        [f"./dataset/ucfTrainTestlist/trainlist03.txt"],
        cfg.n_frames,
        cfg.image_size,
    ),
    cfg.batch_size,
    shuffle=False,
)

trainer.run(train_loader, max_epochs=cfg.epochs)
torch.save(model, f"./checkpoints/ckpt-{datetime.datetime.now()}.pt")

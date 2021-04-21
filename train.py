import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Loss
from torch.utils.data import DataLoader

import wandb
from config import Config
from ucf101 import UCF101
from vivit import ViViT

cfg = Config()
model = ViViT(240, 16, 101, cfg.n_frames).cuda()
model = nn.DataParallel(model)
optimizer = torch.optim.SGD(model.parameters(), lr=cfg.base_lr)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lambda x: (512 ** -0.5)
    * min((x + 1) ** (-0.5), (x + 1) * cfg.warmup_steps ** (-1.5)),
)

# ce_loss_fn = nn.CrossEntropyLoss()
mlsm_loss_fn = nn.MultiLabelSoftMarginLoss()


def onehot_label(class_num: torch.Tensor):
    return F.one_hot(class_num, num_classes=101)


print(sum(p.numel() for p in model.parameters()))


def train_step(engine, batch):
    model.train()
    optimizer.zero_grad()
    video, class_num = batch["video"].cuda(), batch["class"].cuda()
    pred = model(video)
    pred = F.softmax(pred, dim=1)
    # loss = ce_loss_fn(pred, class_num)
    loss = mlsm_loss_fn(pred, onehot_label(class_num))
    loss.backward()
    optimizer.step()
    scheduler.step()
    torch.cuda.empty_cache()
    return loss.item()


trainer = Engine(train_step)


def validation_step(engine, batch):
    model.eval()
    with torch.no_grad():
        video, class_num = batch["video"].cuda(), batch["class"].cuda()
        pred = model(video)
        pred = F.softmax(pred, dim=1)
        torch.cuda.empty_cache()

    return pred, class_num


def mlsm_output_transform(output):
    pred, class_num = output
    onehot_class_num = onehot_label(class_num)
    return pred, onehot_class_num


evaluator = Engine(validation_step)

accuracy_metric = Accuracy()
accuracy_metric.attach(evaluator, "accuracy")

# ce_loss_metric = Loss(ce_loss_fn)
# ce_loss_metric.attach(evaluator, "loss")
mlsm_loss_metric = Loss(mlsm_loss_fn, output_transform=mlsm_output_transform)
mlsm_loss_metric.attach(evaluator, "loss")


@trainer.on(Events.ITERATION_COMPLETED)
def log_training_loss(engine):
    e = engine.state.epoch
    i = engine.state.iteration
    loss = engine.state.output
    print(f"Epoch: {e} / {cfg.epochs} : {i} - Loss: {loss:.5f}")
    wandb.log({"loss": loss})


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(engine):
    state = evaluator.run(train_loader)
    metrics = state.metrics
    loss = metrics["loss"]
    accucary = metrics["accuracy"]
    e = engine.state.epoch
    print(f"Training Results - Loss: {loss:.5f}, Avg accuracy: {accucary:.5f}")
    wandb.log({"train_loss": loss, "train_accuracy": accuracy})


@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(engine):
    state = evaluator.run(val_loader)
    metrics = state.metrics
    loss = metrics["loss"]
    accucary = metrics["accuracy"]
    print(f"Valiation Results - Loss: {loss:.5f}, Avg accuracy: {accucary:.5f}")
    wandb.log({"validation_loss": loss, "validation_accuracy": accuracy})


wandb.init(project="vivit", name=f"vivit-{datetime.datetime.now()}")

# k times cross validation
train_loader = DataLoader(
    UCF101(
        "./dataset/UCF101",
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
        [f"./dataset/ucfTrainTestlist/trainlist03.txt"],
        cfg.n_frames,
        cfg.image_size,
    ),
    cfg.batch_size,
    shuffle=False,
)
trainer.run(train_loader, max_epochs=cfg.epochs)
torch.save(model, f"./checkpoints/ckpt.pt")

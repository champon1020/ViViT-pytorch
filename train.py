import torch
import torch.nn as nn
from iginte.handlers import EarlyStopping
from ignite.contrib.handlers.wandb_logger import WandBLogger
from ignite.engine import (Engine, Events, create_supervised_evaluator,
                           create_supervised_trainer)
from ignite.metrics import Accuracy, Loss
from torch.data.utils import DataLoader

from config import Config
from ucf101 import UCF101
from vivit import ViViT

cfg = Config()
model = ViViT(320, 16, 101, 196).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
loss_fn = nn.CrossEntrypyLoss()

trainer = create_supervised_trainer(model, optimizer, loss_fn)
val_metrics = {"accuracy": Accuracy(), "ce": Loss(loss_fn)}
evaluator = create_supervised_evaluator(model, metrics=val_metrics)


def score_fn(engine: Engine):
    val_loss = engine.state.metrics["ce"]
    return -val_loss


early_stopping_handler = EarlyStopping(
    patience=cfg.es_patience,
    score_function=score_fn,
    trainer=trainer,
)
evaluator.add_event_handler(Events.COMPLETED, early_stopping_handler)


wandb_logger = WandBLogger(
    project="vivit",
    name="vivit",
    config={
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "es_patience": cfg.es_patience,
    },
)

wandb_logger.attach_output_handler(
    trainer,
    event_name=Events.ITERATION_COMPLETED,
    tag="training",
    output_transform=lambda loss: {"loss": loss},
)
wandb_logger.attach_output_handler(
    evaluator,
    event_name=Events.EPOCH_COMPLETED,
    tag="training",
    metric_names=["ce", "accuracy"],
    global_step_transform=lambda *_: trainer.state.iteration,
)


@trainer.on(Events.ITERATION_COMPLETED())
def log_training_loss(trainer):
    print(
        f"Epoch: {trainer.state.epoch} / {cfg.epochs}, Loss: {trainer.state.output:.2f}"
    )


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    evaluator.run(train_loader)
    metrics = evaluator.state.metrics
    print(
        f"Training Results - Epoch: {trainer.state.epoch} Avg accuracy: {metrics['accuracy']:.2f}"
    )


@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    evaluator.run(val_loader)
    metrics = evaluator.state.metrics
    print(
        f"Valiation Results - Epoch: {trainer.state.epoch} Avg accuracy: {metrics['accuracy']:.2f}"
    )


wandb_logger.watch(model)

# k times cross validation
for i in range(3):
    train_loader = DataLoader(
        UCF101(
            "./dataset/UCF101",
            [
                f"./dataset/ucfTrainTestlist/trainlist0{i+1}.txt",
                f"./dataset/ucfTrainTestlist/trainlist0{(i+1)%4+1}.txt",
            ],
        ),
        cfg.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        UCF101(
            "./dataset/UCF101",
            [f"./dataset/ucfTrainTestlist/trainlist0{(i+1)%4+2}.txt"],
        ),
        cfg.batch_size,
        shuffle=False,
    )
    trainer.run(train_loader, max_epochs=cfg.epochs)
    torch.save(model, f"./checkpoints/ckpt-{i+1}.pt")

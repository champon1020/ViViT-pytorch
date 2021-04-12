import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from ignite.handlers import EarlyStopping
from ignite.engine import Engine, Events, create_supervised_evaluator                        
from ignite.metrics import Accuracy, Loss
from torch.utils.data import DataLoader

from config import Config
from ucf101 import UCF101
from vivit import ViViT

cfg = Config()
model = ViViT(240, 16, 101, 1200).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
loss_fn = nn.CrossEntropyLoss()

print(sum(p.numel() for p in model.parameters()))

def train_step(engine, batch):
    model.train()
    optimizer.zero_grad()
    loss = torch.as_tensor(0.0).cuda()
    for b in batch:
        video, class_num = b["video"].cuda(), b["class"].cuda()
        pred = model(video.unsqueeze(0))
        pred = F.softmax(pred, dim=1)
        loss += loss_fn(pred, class_num.unsqueeze(0))
        torch.cuda.empty_cache()
    loss /= len(batch)
    loss.backward()
    optimizer.step()
    return loss.item()

trainer = Engine(train_step)
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

@trainer.on(Events.ITERATION_COMPLETED)
def log_training_loss(trainer):
    print(
        f"Epoch: {trainer.state.epoch} / {cfg.epochs}, Loss: {trainer.state.output:.2f}"
    )
    wandb.log({"loss": trainer.state.output})


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    evaluator.run(train_loader)
    metrics = evaluator.state.metrics
    print(
        f"Training Results - Epoch: {trainer.state.epoch} Avg accuracy: {metrics['accuracy']:.2f}"
    )
    wandb.log({"train_accuracy": metrics["accuracy"]})    


@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    evaluator.run(val_loader)
    metrics = evaluator.state.metrics
    print(
        f"Valiation Results - Epoch: {trainer.state.epoch} Avg accuracy: {metrics['accuracy']:.2f}"
    )
    wandb.log({"validation_accuracy": metrics["accuracy"]})        


wandb.init(project="vivit", name=f"vivit-{datetime.datetime.now()}")

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
        collate_fn=lambda data: data,
    )
    val_loader = DataLoader(
        UCF101(
            "./dataset/UCF101",
            [f"./dataset/ucfTrainTestlist/trainlist0{(i+1)%4+2}.txt"],
        ),
        cfg.batch_size,
        shuffle=False,
        collate_fn=lambda data: data,        
    )
    trainer.run(train_loader, max_epochs=cfg.epochs)
    torch.save(model, f"./checkpoints/ckpt-{i+1}.pt")

import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import wandb
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Loss
from torch.utils.data import DataLoader

from config import Config
from ucf101 import UCF101
from vivit import ViViT

cfg = Config()
model = ViViT(240, 16, 101, cfg.n_frames).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.base_lr)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lambda x: (512 ** -0.5)
    * min((x + 1) ** (-0.5), (x + 1) * cfg.warmup_steps ** (-1.5)),
)
loss_fn = nn.CrossEntropyLoss()

print(sum(p.numel() for p in model.parameters()))


transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.CenterCrop(240),
        torchvision.transforms.ToTensor(),
    ]
)


def load_video(video_path: str):
    video = torchvision.io.read_video(video_path, pts_unit="sec")[0].float()

    # [t, n, h, w]
    interval = video.shape[0] // cfg.n_frames
    video = video.permute(0, 3, 1, 2)
    new_video = torch.zeros(
        (cfg.n_frames, video.shape[1], video.shape[2], video.shape[2])
    )
    for t in range(0, video.shape[0], interval):
        frame = transforms(video[t])
        new_video[t] = frame / 255

    return new_video


def train_step(engine, batch):
    model.train()
    optimizer.zero_grad()
    loss = torch.as_tensor(0.0).cuda()
    for b in batch:
        video_path, class_num = b["video_path"], b["class"].cuda()
        video = load_video(video_path).cuda()
        pred = model(video.unsqueeze(0))
        pred = F.softmax(pred, dim=1)
        loss += loss_fn(pred, class_num.unsqueeze(0))
        torch.cuda.empty_cache()

    loss /= len(batch)
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.item()


trainer = Engine(train_step)


def validation_step(engine, batch):
    model.eval()
    cnt = 0
    loss = torch.as_tensor(0.0).cuda()
    with torch.no_grad():
        for b in batch:
            video_path, class_num = b["video_path"], b["class"].cuda()
            video = load_video(video_path).cuda()
            pred = model(video.unsqueeze(0))
            pred = F.softmax(pred, dim=1)
            loss += loss_fn(pred, class_num.unsqueeze(0))
            if torch.argmax(pred, dim=1) == class_num:
                cnt += 1
            torch.cuda.empty_cache()

    loss /= len(batch)
    return loss.item()


def output_transform(output):
    return output["pred"], output["target"]


evaluator = Engine(validation_step)
metric = Accuracy(output_transform=output_transform)
metric.attach(evaluator, "accuracy")


@trainer.on(Events.ITERATION_COMPLETED)
def log_training_loss(engine):
    e = engine.state.epoch
    i = engine.state.iteration
    loss = engine.state.output
    print(f"Epoch: {e} / {cfg.epochs} : {i} - Loss: {loss:.2f}")
    wandb.log({"loss": loss})


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(engine):
    state = evaluator.run(train_loader)
    metrics = state.metrics
    e = engine.state.epoch
    print(f"Training Results - Epoch: {e} Avg accuracy: {metrics['accuracy']:.2f}")
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
for i in range(1, 4):
    train_loader = DataLoader(
        UCF101(
            "./dataset/UCF101",
            [
                f"./dataset/ucfTrainTestlist/trainlist0{i}.txt",
                f"./dataset/ucfTrainTestlist/trainlist0{i%4+1}.txt",
            ],
        ),
        cfg.batch_size,
        shuffle=True,
        collate_fn=lambda data: data,
    )
    val_loader = DataLoader(
        UCF101(
            "./dataset/UCF101",
            [f"./dataset/ucfTrainTestlist/trainlist0{i%4+2}.txt"],
        ),
        cfg.batch_size,
        shuffle=False,
        collate_fn=lambda data: data,
    )
    trainer.run(train_loader, max_epochs=cfg.epochs)
    torch.save(model, f"./checkpoints/ckpt-{i}.pt")

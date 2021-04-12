import torch
import datetime
import torch.nn as nn
import torch.nn.functional as F
import wandb
import torchvision
from ignite.handlers import EarlyStopping
from ignite.engine import Engine, Events, create_supervised_evaluator                        
from ignite.metrics import Accuracy, Loss
from torch.utils.data import DataLoader

from config import Config
from ucf101 import UCF101
from vivit import ViViT

cfg = Config()
model = ViViT(240, 16, 101, 1800).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.base_lr)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lambda x: (512 ** -0.5) * min((x+1) ** (-0.5), (x+1) * cfg.warmup_steps ** (-1.5)),
)
loss_fn = nn.CrossEntropyLoss()

print(sum(p.numel() for p in model.parameters()))


transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.CenterCrop(240),
    torchvision.transforms.ToTensor(),
])

def load_video(video_path: str):
    video = torchvision.io.read_video(video_path, pts_unit="sec")[0].float()

    # [t, n, h, w]
    video = video.permute(0, 3, 1, 2)
    new_video = torch.zeros((video.shape[0], video.shape[1], video.shape[2], video.shape[2]))
    for t in range(video.shape[0]):
        frame = transforms(video[t])
        new_video[t] = (frame - 127.5) / 127.5

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
val_metrics = {"accuracy": Accuracy(), "ce": Loss(loss_fn)}
evaluator = create_supervised_evaluator(model, metrics=val_metrics)


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

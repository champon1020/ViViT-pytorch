import dataclasses


@dataclasses.dataclass(frozen=True)
class Config:
    epochs = 30
    batch_size = 16
    base_lr = 0.01
    image_size = 112
    n_frames = 16
    warmup_steps = 2000
    multistep_milestones = [10, 20, 30]

import dataclasses


@dataclasses.dataclass(frozen=True)
class Config:
    epochs = 30
    batch_size = 2
    base_lr = 1.0
    warmup_steps = 4000
    n_frames = 16

import dataclasses


@dataclasses.dataclass(frozen=True)
class Config:
    epochs = 30
    batch_size = 16
    base_lr = 2.0
    warmup_steps = 4000
    image_size = 112
    n_frames = 16

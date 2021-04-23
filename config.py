import dataclasses


@dataclasses.dataclass(frozen=True)
class Config:
    epochs = 30
    batch_size = 64
    base_lr = 0.1
    image_size = 112
    n_frames = 16

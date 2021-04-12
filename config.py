import dataclasses


@dataclasses.dataclass(frozen=True)
class Config:
    epochs = 200
    batch_size = 1
    es_patience = 10
    lr = 0.001

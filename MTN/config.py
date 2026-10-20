"""Small, explicit defaults shared by the local MTN smoke tests."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MTNConfig:
    observe_length: int = 15
    predict_length: int = 45
    box_size: int = 4
    speed_size: int = 1
    ego_patches: int = 64
    ped_patches: int = 9
    embedding_size: int = 32
    feed_forward_size: int = 128
    heads: int = 4
    layers: int = 3
    dropout: float = 0.1

    @property
    def sequence_length(self) -> int:
        return self.observe_length + self.predict_length


DEFAULT_CONFIG = MTNConfig()

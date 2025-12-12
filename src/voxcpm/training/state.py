from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainingState:
    """
    Container that mirrors the object returned in the minicpm-audio training
    loop. It holds persistent references to the model, optimizer, scheduler,
    dataloaders and tracker.
    """

    generator: object
    optimizer: object
    scheduler: object
    train_loader: object
    val_loader: object
    tracker: object
    batch_processor: object


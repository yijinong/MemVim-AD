import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)
from torch import nn, optim
from torch.utils.data import DataLoader

from .logging import get_logger

logger = get_logger("Trainer", logging.INFO)


@dataclass(init=False)
class RunnerConfig:
    model: nn.Module
    optimizer: optim.Optimizer
    scheduler: optim.lr_scheduler.LRScheduler
    criterion: nn.Module

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader

    num_epochs: int = 110
    val_interval: int = 10
    ckpt_interval: int = 10
    working_dir: Path = Path("./output")

    resume: bool = False
    load_from: Path | None = None


class Runner:
    """
    Class for running the training and testing of the model
    """

    def __init__(self, config: RunnerConfig) -> None:
        self.logger = logger
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Default config
        self.start_epoch = 0
        self.best_loss = float("inf")

        # Transfer data from config to Runner class
        self.model = config.model.to(self.device)
        self.optimizer = config.optimizer
        self.scheduler = config.scheduler
        self.criterion = config.criterion

        self.train_loader = config.train_loader
        self.val_loader = config.val_loader
        self.test_loader = config.test_loader

        self.num_epochs = config.num_epochs
        self.val_interval = config.val_interval
        self.ckpt_interval = config.ckpt_interval
        self.working_dir = config.working_dir

        if config.resume:
            self._load_ckpt(config.load_from)
        self._create_dirs()

    def __enter__(self) -> "Runner":
        self.progress.start()
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.progress.stop()
        return

    def _create_dirs(self) -> None:
        (self.working_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.working_dir / "logs").mkdir(parents=True, exist_ok=True)

    def _load_ckpt(self, ckpt_file: Path | None = None) -> None:
        if ckpt_file is None:
            ckpt_dir = self.working_dir / "checkpoints"
            ckpts = sorted(
                ckpt_dir.glob("epoch*.pth"),
                key=lambda f: f.stat().st_mtime,
                reverse=True,
            )
            ckpt_file = ckpts[0] if ckpts else None

        if ckpt_file is None:
            self.logger.warning("No checkpoint file found to load.")
            raise FileNotFoundError(
                "No checkpoint file found in the checkpoints directory."
            )

        self.logger.info("Loading checkpoint from %s", ckpt_file)

        ckpt = torch.load(ckpt_file, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.start_epoch = ckpt["epoch"] + 1
        self.best_loss = ckpt.get("best_loss", float("inf"))

        self.logger.info("Checkpoint loaded successfully from %s", ckpt_file)

    def _save_ckpt(self, epoch: int) -> None:
        ckpt = {
            "model_state": getattr(self.model, "module", self.model).state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epoch": epoch,
            "best_loss": self.best_loss,
        }

        ckpt_path = self.working_dir / "checkpoints" / f"epoch_{epoch}.pth"
        torch.save(ckpt, ckpt_path)
        self.logger.info(f"Epoch: {epoch + 1}: Checkpoint saved at {ckpt_path}")

    def _save_best(self) -> None:
        ckpt = {"model_state": getattr(self.model, "module", self.model).state_dict()}
        ckpt_path = self.working_dir / "checkpoints" / "best_model.pth"
        torch.save(ckpt, ckpt_path)
        self.logger.info(f"Best model saved at {ckpt_path}")

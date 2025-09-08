import argparse
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.optim as optim
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .logging import get_logger
# from .loss_func import Contrastive_Loss, PrototypeAlignmentLoss
from .mvtec_data.load_dataset import MVTecADJSONDataModule, MVTecADJSONDataset
from .model.memvim import MemVim
from .rich_logging import rich_interface

logger = get_logger("TRAINER", level=logging.INFO)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MVTecAD Hierarchical Prototype Learning using Vision Mamba"
    )
    parser.add_argument(
        "--root_dir", type=str, required=True, help="Path to MVTecAD dataset"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["all", "eval"],
        default="all",
        help="Training mode: all categories, or evaluation",
    )
    parser.add_argument(
        "--save_dir", type=str, default="./models", help="Directory to save models"
    )
    parser.add_argument(
        "--model_path", type=str, default=None, help="Path to model for evaluation mode"
    )

    # Training hyperparameters
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--k_coarse", type=int, default=10)
    parser.add_argument("--k_fine", type=int, default=5)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--feedback_ratio", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="auto")

    # Dynamic loss weight parameters
    parser.add_argument(
        "--lambda_con_init",
        type=float,
        default=1.0,
        help="Initial contrastive loss weight",
    )
    parser.add_argument(
        "--lambda_proto_init",
        type=float,
        default=0.1,
        help="Initial prototype loss weight",
    )
    parser.add_argument(
        "--lambda_reg_init",
        type=float,
        default=0.01,
        help="Initial split regularization loss weight",
    )
    parser.add_argument(
        "--lambda_con_decay_rate",
        type=float,
        default=0.95,
        help="Decay rate for contrastive loss weight",
    )
    parser.add_argument(
        "--lambda_proto_increase_factor",
        type=float,
        default=1.1,
        help="Increase factor for prototype loss weight after splits",
    )

    # Learning rate scheduler parameters
    parser.add_argument(
        "--scheduler_type",
        type=str,
        choices=["step", "cosine", "none"],
        default="step",
        help="Type of learning rate scheduler",
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=30,
        help="Step size for StepLR scheduler",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help="Gamma for StepLR scheduler",
    )
    parser.add_argument(
        "--cosine_t_max",
        type=int,
        default=None,
        help="T_max for CosineAnnealingLR scheduler (defaults to num_epochs)",
    )

    return parser.parse_args()

def create_mvtecad_memvim_setup(
    root_dir: str,
    category: str,
    img_size: int = 512,
    batch_size: int = 32,
    feedback_ratio: float = 0.1,
    augment_train: bool = True,
    load_masks: bool = False,
):
    data_module = MVTecADJSONDataModule(
        train_json="/home/yijin/projects/MemVim-AD/data/mvtec-ad/train.json",
        test_json="/home/yijin/projects/MemVim-AD/data/mvtec-ad/test.json",
        root_dir=root_dir,
        category=category,
        img_size=img_size,
        batch_size=batch_size,
        augment_train=augment_train,
        load_masks=load_masks,
    )
    train_loader = data_module.get_train_dataloader()
    anomaly_loader = data_module.get_anomaly_feedback_dataloader(feedback_ratio=feedback_ratio)
    test_loader = data_module.get_combined_test_dataloader()

    return train_loader, anomaly_loader, test_loader

def train_all_categories(root_dir:str, save_dir: str = "./models", **training_kwargs):
    logger.info(f"Training universal model on all categories in {root_dir}")

    # Category name is the root folder name
    data_root = Path(root_dir)
    categories = [p.name for p in data_root.iterdir() if p.is_dir()]
    print(f"Found categories: {categories}")

    # Set up tensorboard writer
    writer = SummaryWriter(log_dir=str(Path(save_dir) / "logs"))
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Load data from all categories at once
    logger.info(f"Loading data from all categories")
    train_loader, anomaly_loader, test_loader = create_mvtecad_memvim_setup(
        train_json="/home/yijin/projects/MemVim-AD/data/mvtec-ad/train.json",
        test_json="/home/yijin/projects/MemVim-AD/data/mvtec-ad/test.json",
        root_dir=root_dir,
        category=None,  # No category filter means load all categories
        img_size=training_kwargs.get("img_size", 224),
        batch_size=training_kwargs.get("batch_size", 32),
        feedback_ratio=training_kwargs.get("feedback_ratio", 0.1),
        augment_train=True,
        load_masks=True,
    )

    model = MemVim(
        k_coarse=training_kwargs.get("k_coarse", 10),
        k_fine=training_kwargs.get("k_fine", 5),
        d_model=training_kwargs.get("d_model", 256),
        n_layers=training_kwargs.get("n_layers", 4),
        device=training_kwargs.get("device", "cuda"),
    ).to(training_kwargs.get("device", "cuda"))

    # loss functions
    optimizer = optim.Adam(model.parameters(), lr=training_kwargs.get("learning_rate", 2e-4))

    if training_kwargs.get("scheduler_type", "step") == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=training_kwargs.get("step_size", 30),
            gamma=training_kwargs.get("gamma", 0.1),
        )
    elif training_kwargs.get("scheduler_type") == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=training_kwargs.get("cosine_t_max", training_kwargs.get("num_epochs", 50)),
        )
    else:
        scheduler = None

    # Dynamic loss weights
    lambda_con = training_kwargs.get("lambda_con_init", 1.0)
    lambda_proto = training_kwargs.get("lambda_proto_init", 0.1)
    lambda_reg = training_kwargs.get("lambda_reg_init", 0.01)
    lambda_con_decay_rate = training_kwargs.get("lambda_con_decay_rate", 0.95)
    lambda_proto_increase_factor = training_kwargs.get("lambda_proto_increase_factor", 1.1)

    best_auroc = 0.0
    best_model_path = None

    # Training loop
    for epoch in range(training_kwargs.get("num_epochs", 50)):
        model.train()
        total_loss = 0
        start_time = time.time()

        # Training with normal samples
        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"].to(training_kwargs.get("device", "cuda"))
            
            optimizer.zero_grad()
            loss = model.compute_loss(images)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        # Training with anomaly feedback
        if anomaly_loader is not None:
            for batch_idx, batch in enumerate(anomaly_loader):
                images = batch["image"].to(training_kwargs.get("device", "cuda"))
                labels = batch["label"].to(training_kwargs.get("device", "cuda"))
                
                optimizer.zero_grad()
                loss = model.compute_loss(images, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()

        avg_loss = total_loss / (len(train_loader) + (len(anomaly_loader) if anomaly_loader else 0))
        
        # Evaluate on test set
        model.eval()
        all_scores = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(training_kwargs.get("device", "cuda"))
                labels = batch["label"].to(training_kwargs.get("device", "cuda"))
                
                scores = model.get_anomaly_score(images)
                
                all_scores.extend(scores.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        auroc = roc_auc_score(all_labels, all_scores)
        ap = average_precision_score(all_labels, all_scores)

        # Log metrics
        writer.add_scalar("training/loss", avg_loss, epoch)
        writer.add_scalar("training/auroc", auroc, epoch)
        writer.add_scalar("training/ap", ap, epoch)

        # Update best model
        if auroc > best_auroc:
            best_auroc = auroc
            # Save best model
            best_model_path = Path(save_dir) / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'auroc': auroc,
                'ap': ap,
            }, str(best_model_path))

        # Update learning rate
        if scheduler is not None:
            scheduler.step()

        # Update loss weights
        lambda_con *= lambda_con_decay_rate
        model.update_loss_weights(lambda_con=lambda_con, lambda_proto=lambda_proto, lambda_reg=lambda_reg)

        # Log progress
        logger.info(f"Epoch {epoch+1}/{training_kwargs.get('num_epochs', 50)}, "
                    f"Loss: {avg_loss:.4f}, AUROC: {auroc:.4f}, AP: {ap:.4f}")

    # Store results
    results = {
        'best_auroc': best_auroc,
        'final_ap': ap,
        'best_model_path': str(best_model_path)
    }
    
    # Save final model
    final_model_path = Path(save_dir) / "final_model.pth"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'auroc': auroc,
        'ap': ap,
    }, str(final_model_path))

    writer.close()
    return results

def main():
    rich_interface.print_banner(
        "🧠 MVTecAD Hierarchical Prototype Learning",
        "Advanced Anomaly Detection with Mamba Architecture",
    )

    args = parse_args()

    # Set device
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    rich_interface.print_info(f"Using device: {args.device}", "System Configuration")

    
    train_all_categories(
        root_dir=args.root_dir,
        save_dir=args.save_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=args.device,
    )

if __name__ == "__main__":
    main()


import argparse
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.optim as optim
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .logging import get_logger
from .loss_func import ContrastiveLoss, PrototypeAlignmentLoss
from .model.hierarchy_proto import HierarchicalPrototypeMemory
from .model.mamba_model import MambaFeatureExtractor
from .mvtec_data.dataset import MVTecADDataModule, MVTecADDataset
from .rich_logging import rich_interface

logger = get_logger("Training", logging.DEBUG)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MVTecAD Hierarchical Prototype Learning"
    )
    parser.add_argument(
        "--root_dir", type=str, required=True, help="Path to MVTecAD dataset"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "all", "eval"],
        default="single",
        help="Training mode: single category, all categories, or evaluation",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="bottle",
        help="Category to train (for single mode)",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=None,
        help="List of categories to train (for all mode)",
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


class HierarchicalPrototypeLearner:
    """Main training class for hierarchical prototype learning"""

    def __init__(
        self,
        feature_extractor: MambaFeatureExtractor,
        k_coarse: int = 10,
        k_fine: int = 5,
        update_interval: int = 10,
        learning_rate: float = 1e-4,
        temperature: float = 0.07,
        device: str = "cuda",
        log_dir: Optional[str] = None,
        # Dynamic loss weight parameters
        lambda_con_init: float = 1.0,
        lambda_proto_init: float = 0.1,
        lambda_reg_init: float = 0.01,
        lambda_con_decay_rate: float = 0.95,
        lambda_proto_increase_factor: float = 1.1,
        # Learning rate scheduler parameters
        scheduler_type: str = "step",  # "step", "cosine", or "none"
        step_size: int = 30,
        gamma: float = 0.1,
        cosine_t_max: Optional[int] = None,  # Will be set to num_epochs if None
    ):
        self.device = device
        self.feature_extractor = feature_extractor.to(device)
        self.k_coarse = k_coarse
        self.k_fine = k_fine
        self.update_interval = update_interval

        # Initialize prototype memory
        # Get d_model from the feature extractor (it's stored as a direct attribute)
        feature_dim = feature_extractor.d_model

        self.prototype_memory = HierarchicalPrototypeMemory(
            feature_dim, k_coarse, k_fine
        )

        # Losses
        self.contrastive_loss = ContrastiveLoss(temperature)
        self.prototype_loss = PrototypeAlignmentLoss()

        # Dynamic loss weights
        self.lambda_con = lambda_con_init
        self.lambda_proto = lambda_proto_init
        self.lambda_reg = lambda_reg_init
        self.lambda_con_init = lambda_con_init
        self.lambda_proto_init = lambda_proto_init
        self.lambda_con_decay_rate = lambda_con_decay_rate
        self.lambda_proto_increase_factor = lambda_proto_increase_factor

        # Track representation learning stability
        self.contrastive_loss_history = []
        self.representation_stable_threshold = (
            0.01  # Threshold for stable representation learning
        )

        # Optimizer
        self.optimizer = optim.Adam(
            task = progress.add_task(
                "Extracting features", total=len(normal_dataloader), loss=0.0
            )

            with torch.no_grad():
                for _, batch in enumerate(normal_dataloader):
                    images = batch["image"].to(self.device)
                    features = self.extract_features(images)
                    all_features.append(features)

                    progress.update(task, advance=1, loss=0.0)

        all_features = torch.cat(all_features, dim=0)
        self.prototype_memory.initialize_prototypes(all_features)

        # Store features in memory buffer
        self.memory_buffer = all_features
        rich_interface.print_info(
            f"Stored {len(all_features)} features in memory buffer",
            "Initialization Complete",
        )

    def train_epoch(
        self,
        dataloader: DataLoader,
        anomaly_dataloader: Optional[DataLoader] = None,
        epoch: int = 0,
    ):
        """Train for one epoch"""
        self.feature_extractor.train()
        self.training = True  # Set training flag for rich interface

        total_contrastive_loss = 0.0
        total_prototype_loss = 0.0
        total_split_reg_loss = 0.0
        total_loss_value = 0.0
        num_batches = 0
        split_occurred = False

        # Track epoch start time for rich logging
        epoch_start_time = time.time()

        for batch_idx, batch in enumerate(dataloader):
            images = batch["image"].to(self.device)

            # Extract features
            features = self.feature_extractor(images)

            # Compute individual losses
            contrastive_loss = self.contrastive_loss(features)
            prototype_loss = self.prototype_loss(features, self.prototype_memory)
            split_reg_loss = self.compute_split_regularization_loss()

            # Compute total loss with dynamic weights
            total_loss = (
                self.lambda_con * contrastive_loss
                + self.lambda_proto * prototype_loss
                + self.lambda_reg * split_reg_loss
            )

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # Accumulate losses for logging
            total_contrastive_loss += contrastive_loss.item()
            total_prototype_loss += prototype_loss.item()
            total_split_reg_loss += split_reg_loss.item()
            total_loss_value += total_loss.item()
            num_batches += 1
            self.global_step += 1

            # Log to TensorBoard
            if self.writer and batch_idx % 10 == 0:
                self.writer.add_scalar(
                    "Train/Contrastive_Loss", contrastive_loss.item(), self.global_step
                )
                self.writer.add_scalar(
                    "Train/Prototype_Loss", prototype_loss.item(), self.global_step
                )
                self.writer.add_scalar(
                    "Train/Split_Reg_Loss", split_reg_loss.item(), self.global_step
                )
                self.writer.add_scalar(
                    "Train/Total_Loss", total_loss.item(), self.global_step
                )

                # Log learning rate
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.writer.add_scalar(
                    "Train/Learning_Rate", current_lr, self.global_step
                )

            # Log batch losses using rich console instead of standard logger
            if batch_idx % 50 == 0:
                batch_losses = {
                    "total_loss": total_loss.item(),
                    "contrastive_loss": contrastive_loss.item(),
                    "prototype_loss": prototype_loss.item(),
                    "split_reg_loss": split_reg_loss.item(),
                }

                lambda_weights = {
                    "lambda_con": self.lambda_con,
                    "lambda_proto": self.lambda_proto,
                    "lambda_reg": self.lambda_reg,
                }

                rich_interface.log_batch_loss(
                    epoch + 1, batch_idx, len(dataloader), batch_losses, lambda_weights
                )

        # Process anomaly feedback if available
        if anomaly_dataloader is not None:
            split_occurred = self.process_anomaly_feedback(anomaly_dataloader)

        # Update contrastive loss history for stability tracking
        avg_contrastive_loss = total_contrastive_loss / num_batches
        self.contrastive_loss_history.append(avg_contrastive_loss)
        if len(self.contrastive_loss_history) > 20:  # Keep only recent history
            self.contrastive_loss_history.pop(0)

        # Update dynamic loss weights
        self.update_loss_weights(split_occurred)

        avg_prototype_loss = total_prototype_loss / num_batches
        avg_split_reg_loss = total_split_reg_loss / num_batches
        avg_total_loss = total_loss_value / num_batches

        # Calculate epoch duration
        epoch_duration = time.time() - epoch_start_time

        # Log epoch summary using rich console
        epoch_losses = {
            "total_loss": avg_total_loss,
            "contrastive_loss": avg_contrastive_loss,
            "prototype_loss": avg_prototype_loss,
            "split_reg_loss": avg_split_reg_loss,
        }
        current_lr = self.optimizer.param_groups[0]["lr"]
        rich_interface.log_epoch_summary(
            epoch + 1, epoch_losses, current_lr, epoch_duration
        )

        # Log epoch averages to TensorBoard
        if self.writer:
            self.writer.add_scalar(
                "Epoch/Avg_Contrastive_Loss", avg_contrastive_loss, epoch
            )
            self.writer.add_scalar(
                "Epoch/Avg_Prototype_Loss", avg_prototype_loss, epoch
            )
            self.writer.add_scalar(
                "Epoch/Avg_Split_Reg_Loss", avg_split_reg_loss, epoch
            )
            self.writer.add_scalar("Epoch/Avg_Total_Loss", avg_total_loss, epoch)

        return (
            avg_contrastive_loss,
            avg_prototype_loss,
            avg_split_reg_loss,
            avg_total_loss,
        )

    def process_anomaly_feedback(self, anomaly_dataloader: DataLoader) -> bool:
        """Process anomaly feedback and split prototypes if needed"""
        logger.info("Processing anomaly feedback...")

        self.feature_extractor.eval()
        high_score_threshold = 0.8  # Threshold for high anomaly scores
        splits_performed = False

        with torch.no_grad():
            for batch in anomaly_dataloader:
                images = batch["image"].to(self.device)
                features = self.extract_features(images)
                anomaly_scores = self.prototype_memory.compute_anomaly_score(features)

                # Find high-scoring anomalies
                high_score_mask = anomaly_scores > high_score_threshold
                if high_score_mask.sum() > 0:
                    high_score_features = features[high_score_mask]

                    # For each high-scoring anomaly, find affected prototypes and split
                    for feature in high_score_features:
                        feature = feature.unsqueeze(0)

                        # Find closest coarse prototype
                        if self.prototype_memory.coarse_prototypes is not None:
                            coarse_distances = torch.cdist(
                                feature, self.prototype_memory.coarse_prototypes
                            )
                            closest_coarse_idx = torch.argmin(
                                coarse_distances, dim=1
                            ).item()

                            # Find closest fine prototype
                            if (
                                closest_coarse_idx
                                in self.prototype_memory.fine_prototypes
                            ):
                                fine_prototypes = self.prototype_memory.fine_prototypes[
                                    closest_coarse_idx
                                ]
                                fine_distances = torch.cdist(feature, fine_prototypes)
                                closest_fine_idx = torch.argmin(
                                    fine_distances, dim=1
                                ).item()

                                # Split the prototype
                                self.prototype_memory.split_prototype(
                                    int(closest_coarse_idx),
                                    int(closest_fine_idx),
                                    feature,
                                )
                                splits_performed = True

        # Log split information
        if self.writer:
            self.writer.add_scalar(
                "Anomaly_Feedback/Splits_Performed",
                1.0 if splits_performed else 0.0,
                self.global_step,
            )

        return splits_performed

    def compute_anomaly_scores(self, test_dataloader: DataLoader) -> List[float]:
        """Compute anomaly scores for test data"""
        self.feature_extractor.eval()
        all_scores = []

        with torch.no_grad():
            for batch in test_dataloader:
                images = batch["image"].to(self.device)
                features = self.extract_features(images)
                batch_scores = self.prototype_memory.compute_anomaly_score(features)
                all_scores.extend(batch_scores.cpu().numpy().tolist())

        return all_scores

    def evaluate_on_test(self, test_dataloader: DataLoader) -> Dict[str, Any]:
        """Evaluate model performance on test set and return metrics"""

        # Compute anomaly scores
        scores = self.compute_anomaly_scores(test_dataloader)

        # Get ground truth labels
        labels = []
        for batch in test_dataloader:
            labels.extend(batch["label"].numpy().tolist())

        # Compute metrics
        auroc = roc_auc_score(labels, scores)
        auprc = average_precision_score(labels, scores)

        return {
            "auroc": float(auroc),
            "auprc": float(auprc),
            "num_samples": len(scores),
            "num_normal": sum(1 for label in labels if label == 0),
            "num_anomaly": sum(1 for label in labels if label == 1),
        }

    def train(
        self,
        normal_dataloader: DataLoader,
        num_epochs: int,
        anomaly_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None,
    ):
        """Full training pipeline with validation every 10 epochs"""
        rich_interface.print_info(
            "Starting hierarchical prototype learning training...", "Training Start"
        )

        # Initialize cosine scheduler if needed (now we know num_epochs)
        if self.scheduler_type == "cosine":
            t_max = self.cosine_t_max or num_epochs
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=t_max)

        # Initialize prototypes
        self.initialize_prototypes(normal_dataloader)

        # Training loop with rich progress bars
        with rich_interface.create_epoch_progress(num_epochs) as epoch_progress:
            epoch_task = epoch_progress.add_task("Training Progress", total=num_epochs)

            for epoch in range(num_epochs):
                # Update epoch progress
                epoch_progress.update(epoch_task, advance=1)

                (contrastive_loss, prototype_loss, split_reg_loss, total_loss) = (
                    self.train_epoch(
                        normal_dataloader,
                        anomaly_dataloader
                        if (epoch + 1) % self.update_interval == 0
                        else None,
                        epoch,
                    )
                )

                # Step the learning rate scheduler
                if self.scheduler is not None:
                    self.scheduler.step()

                # Get current learning rate for logging
                current_lr = self.optimizer.param_groups[0]["lr"]

                # Create metrics dictionary for rich display
                epoch_metrics = {
                    "total_loss": total_loss,
                    "contrastive_loss": contrastive_loss,
                    "prototype_loss": prototype_loss,
                    "split_reg_loss": split_reg_loss,
                    "learning_rate": current_lr,
                }

                # Update best metrics tracking
                for metric_name, value in epoch_metrics.items():
                    if "loss" in metric_name.lower():
                        # For losses, lower is better
                        if (
                            metric_name not in self.best_metrics
                            or value < self.best_metrics[metric_name]
                        ):
                            self.best_metrics[metric_name] = value
                    else:
                        # For other metrics like learning rate, just track current
                        self.best_metrics[metric_name] = value

                # Display metrics table every 10 epochs or on last epoch
                if epoch % 10 == 0 or epoch == num_epochs - 1:
                    metrics_table = rich_interface.update_metrics_table(
                        epoch + 1, epoch_metrics
                    )
                    rich_interface.console.print(metrics_table)

                # Periodic validation every 10 epochs
                if test_dataloader is not None and epoch % 10 == 0:
                    rich_interface.print_info(
                        f"Validating model (Epoch {epoch + 1})...", "Validation"
                    )
                    eval_results = self.evaluate_on_test(test_dataloader)

                    # Update epoch metrics with evaluation results
                    epoch_metrics.update(
                        {"auroc": eval_results["auroc"], "auprc": eval_results["auprc"]}
                    )

                    # Update best metrics with evaluation results
                    for metric_name in ["auroc", "auprc"]:
                        # For AUROC/AUPRC, higher is better
                        if (
                            metric_name not in self.best_metrics
                            or epoch_metrics[metric_name]
                            > self.best_metrics[metric_name]
                        ):
                            self.best_metrics[metric_name] = epoch_metrics[metric_name]

                    # Log evaluation results to TensorBoard
                    if self.writer:
                        self.writer.add_scalar(
                            "Eval/AUROC", eval_results["auroc"], epoch
                        )
                        self.writer.add_scalar(
                            "Eval/AUPRC", eval_results["auprc"], epoch
                        )

                    # Display evaluation results with rich
                    rich_interface.print_evaluation_results(eval_results)

                # Log prototype memory statistics
                if self.writer:
                    num_coarse = (
                        len(self.prototype_memory.coarse_prototypes)
                        if self.prototype_memory.coarse_prototypes is not None
                        else 0
                    )
                    num_fine_total = sum(
                        len(fine_protos)
                        for fine_protos in self.prototype_memory.fine_prototypes.values()
                    )

                    self.writer.add_scalar("Prototypes/Num_Coarse", num_coarse, epoch)
                    self.writer.add_scalar(
                        "Prototypes/Num_Fine_Total", num_fine_total, epoch
                    )

                    # Log average number of fine prototypes per coarse prototype
                    if num_coarse > 0:
                        avg_fine_per_coarse = num_fine_total / num_coarse
                        self.writer.add_scalar(
                            "Prototypes/Avg_Fine_Per_Coarse", avg_fine_per_coarse, epoch
                        )

                    # Log learning rate
                    self.writer.add_scalar("Optimizer/Learning_Rate", current_lr, epoch)

                    # Display prototype stats table every 20 epochs
                    if epoch % 20 == 0 or epoch == num_epochs - 1:
                        proto_stats = {
                            "num_coarse": num_coarse,
                            "num_fine_total": num_fine_total,
                        }
                        proto_table = rich_interface.create_prototype_stats_table(
                            proto_stats
                        )
                        rich_interface.console.print(proto_table)

        rich_interface.print_info("Training completed!", "Success")

        # Close TensorBoard writer
        if self.writer:
            self.writer.close()


# Integration with the hierarchical prototype learning pipeline
def create_mvtecad_training_setup(
    root_dir: str,
    category: str,
    img_size: int = 224,
    batch_size: int = 32,
    feedback_ratio: float = 0.1,
):
    """
    Create complete training setup for MVTecAD with hierarchical prototype learning

    Args:
        root_dir: Path to MVTecAD dataset
        category: Category to train on
        img_size: Image size
        batch_size: Batch size
        feedback_ratio: Ratio of anomaly data to use for feedback

    Returns:
        Tuple of (normal_dataloader, anomaly_feedback_dataloader, test_dataloader)
    """
    # Create data module
    data_module = MVTecADDataModule(
        root_dir=root_dir,
        category=category,
        img_size=img_size,
        batch_size=batch_size,
        augment_train=True,
        load_masks=False,  # Masks not needed for prototype learning
    )

    # Get dataloaders
    normal_dataloader = data_module.get_train_dataloader()
    anomaly_feedback_dataloader = data_module.get_anomaly_feedback_dataloader(
        feedback_ratio=feedback_ratio
    )
    test_dataloader = data_module.get_combined_test_dataloader()

    logger.info(f"Created training setup for {category}:")
    logger.info(f"  Normal training batches: {len(normal_dataloader)}")
    logger.info(f"  Anomaly feedback batches: {len(anomaly_feedback_dataloader)}")
    logger.info(f"  Test batches: {len(test_dataloader)}")

    return normal_dataloader, anomaly_feedback_dataloader, test_dataloader


def train_single_category(
    root_dir: str, category: str, save_dir: str = "./models", **training_kwargs
):
    """
    Train hierarchical prototype learning on a single MVTecAD category

    Args:
        root_dir: Path to MVTecAD dataset
        category: Category to train on
        save_dir: Directory to save models
        **training_kwargs: Additional arguments for training
    """

    # Start rich training session
    rich_interface.start_training_session(
        category, training_kwargs.get("num_epochs", 50), training_kwargs
    )

    # Setup data
    normal_loader, anomaly_feedback_loader, test_loader = create_mvtecad_training_setup(
        root_dir=root_dir,
        category=category,
        img_size=training_kwargs.get("img_size", 224),
        batch_size=training_kwargs.get("batch_size", 32),
        feedback_ratio=training_kwargs.get("feedback_ratio", 0.1),
    )

    # Display dataset info
    dataset_info = {
        "Normal Training": len(normal_loader) * training_kwargs.get("batch_size", 32),
        "Anomaly Feedback": len(anomaly_feedback_loader)
        * training_kwargs.get("batch_size", 32),
        "Test (Combined)": len(test_loader) * training_kwargs.get("batch_size", 32),
    }
    rich_interface.print_dataset_info(dataset_info)

    # Create feature extractor
    feature_extractor = MambaFeatureExtractor(
        input_dim=3,
        d_model=training_kwargs.get("d_model", 256),
        n_layers=training_kwargs.get("n_layers", 4),
        patch_size=training_kwargs.get("patch_size", 16),
        img_size=training_kwargs.get("img_size", 224),
    )

    # Create learner
    log_dir = Path(save_dir) / f"{category}_logs"
    learner = HierarchicalPrototypeLearner(
        feature_extractor=feature_extractor,
        k_coarse=training_kwargs.get("k_coarse", 10),
        k_fine=training_kwargs.get("k_fine", 5),
        update_interval=training_kwargs.get("update_interval", 10),
        learning_rate=training_kwargs.get("learning_rate", 1e-4),
        device=training_kwargs.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        ),
        log_dir=str(log_dir),
        # Dynamic loss weight parameters
        lambda_con_init=training_kwargs.get("lambda_con_init", 1.0),
        lambda_proto_init=training_kwargs.get("lambda_proto_init", 0.1),
        lambda_reg_init=training_kwargs.get("lambda_reg_init", 0.01),
        lambda_con_decay_rate=training_kwargs.get("lambda_con_decay_rate", 0.95),
        lambda_proto_increase_factor=training_kwargs.get(
            "lambda_proto_increase_factor", 1.1
        ),
        # Learning rate scheduler parameters
        scheduler_type=training_kwargs.get("scheduler_type", "step"),
        step_size=training_kwargs.get("step_size", 30),
        gamma=training_kwargs.get("gamma", 0.1),
        cosine_t_max=training_kwargs.get("cosine_t_max", None),
    )

    # Train the model with validation every 10 epochs
    learner.train(
        normal_dataloader=normal_loader,
        num_epochs=training_kwargs.get("num_epochs", 50),
        anomaly_dataloader=anomaly_feedback_loader,
        test_dataloader=test_loader,
    )

    # Evaluate on test set
    rich_interface.print_info("Computing test scores...", "Evaluation")
    test_scores = learner.compute_anomaly_scores(test_loader)

    # Save model
    save_path = Path(save_dir) / f"{category}_hierarchical_prototype_model.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "feature_extractor": learner.feature_extractor.state_dict(),
            "prototype_memory": {
                "coarse_prototypes": learner.prototype_memory.coarse_prototypes,
                "fine_prototypes": learner.prototype_memory.fine_prototypes,
                "feature_dim": learner.prototype_memory.feature_dim,
                "k_coarse": learner.prototype_memory.k_coarse,
                "k_fine": learner.prototype_memory.k_fine,
            },
            "category": category,
            "test_scores": test_scores,
        },
        save_path,
    )

    # Print training summary
    final_metrics = {"total_loss": 0.0}  # You could extract final loss from learner
    rich_interface.print_training_summary(category, final_metrics, str(save_path))

    return learner, test_scores


def train_all_categories(
    root_dir: str,
    categories: Optional[List[str]] = None,
    save_dir: str = "./models",
    **training_kwargs,
):
    """
    Train hierarchical prototype learning on all or selected MVTecAD categories

    Args:
        root_dir: Path to MVTecAD dataset
        categories: List of categories to train (None for all)
        save_dir: Directory to save models
        parallel: Whether to train categories in parallel (not implemented yet)
        **training_kwargs: Training hyperparameters
    """
    if categories is None:
        categories = MVTecADDataset.CATEGORIES

    results = {}

    logger.info(f"Training on {len(categories)} categories: {categories}")

    for i, category in enumerate(categories, 1):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Training Category {i}/{len(categories)}: {category}")
        logger.info(f"{'=' * 60}")

        try:
            learner, test_scores = train_single_category(
                root_dir=root_dir,
                category=category,
                save_dir=save_dir,
                **training_kwargs,
            )
            results[category] = {
                "learner": learner,
                "test_scores": test_scores,
                "status": "success",
            }
            logger.info(f"✅ Successfully trained {category}")

        except (RuntimeError, ValueError, FileNotFoundError) as e:
            logger.error(f"❌ Failed to train {category}: {str(e)}")
            results[category] = {
                "learner": None,
                "test_scores": None,
                "status": "failed",
                "error": str(e),
            }

    # Print summary
    successful = sum(1 for r in results.values() if r["status"] == "success")
    logger.info(f"\n{'=' * 60}")
    logger.info(
        f"Training Summary: {successful}/{len(categories)} categories successful"
    )
    logger.info(f"{'=' * 60}")

    for category, result in results.items():
        status = "✅" if result["status"] == "success" else "❌"
        logger.info(f"{status} {category}")

    return results


def evaluate_category(
    model_path: str,
    root_dir: str,
    category: str,
    img_size: int = 224,
    batch_size: int = 32,
) -> Dict:
    """
    Evaluate a trained model on MVTecAD test data

    Args:
        model_path: Path to saved model
        root_dir: Path to MVTecAD dataset
        category: Category to evaluate
        img_size: Image size
        batch_size: Batch size

    Returns:
        Dictionary with evaluation results
    """

    # Load model
    checkpoint = torch.load(model_path, map_location="cpu")

    # Setup data
    data_module = MVTecADDataModule(
        root_dir=root_dir,
        category=category,
        img_size=img_size,
        batch_size=batch_size,
        augment_train=False,
        load_masks=False,
    )

    test_loader = data_module.get_combined_test_dataloader()

    # Reconstruct model (you'll need to adapt this based on your implementation)
    feature_extractor = MambaFeatureExtractor(
        input_dim=3,
        d_model=256,  # Should match training config
        n_layers=4,
        patch_size=16,
        img_size=img_size,
    )
    feature_extractor.load_state_dict(checkpoint["feature_extractor"])

    # Create learner first to get the prototype memory initialized
    device = "cuda" if torch.cuda.is_available() else "cpu"
    learner = HierarchicalPrototypeLearner(
        feature_extractor=feature_extractor,
        k_coarse=checkpoint["prototype_memory"]["k_coarse"],
        k_fine=checkpoint["prototype_memory"]["k_fine"],
        device=device,
    )

    # Load prototype memory state and move to correct device
    learner.prototype_memory.coarse_prototypes = checkpoint["prototype_memory"][
        "coarse_prototypes"
    ].to(device)

    # Load fine prototypes and move to correct device
    fine_prototypes = checkpoint["prototype_memory"]["fine_prototypes"]
    for key, value in fine_prototypes.items():
        fine_prototypes[key] = value.to(device)
    learner.prototype_memory.fine_prototypes = fine_prototypes

    # Compute scores
    scores = learner.compute_anomaly_scores(test_loader)

    # Get ground truth labels
    labels = []
    for batch in test_loader:
        labels.extend(batch["label"].numpy().tolist())

    # Compute metrics
    auroc = roc_auc_score(labels, scores)
    auprc = average_precision_score(labels, scores)

    results = {
        "category": category,
        "auroc": auroc,
        "auprc": auprc,
        "num_samples": len(scores),
        "num_normal": sum(1 for label in labels if label == 0),
        "num_anomaly": sum(1 for label in labels if label == 1),
    }

    # Use rich interface for beautiful evaluation results
    rich_interface.print_evaluation_results(results)

    return results


# Main execution
def main() -> None:
    # Print rich banner
    rich_interface.print_banner(
        "🧠 MVTecAD Hierarchical Prototype Learning",
        "Advanced Anomaly Detection with Mamba Architecture",
    )

    args = parse_args()

    # Set device
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    rich_interface.print_info(f"Using device: {args.device}", "System Configuration")

    # Training hyperparameters
    training_kwargs = {
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "k_coarse": args.k_coarse,
        "k_fine": args.k_fine,
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "feedback_ratio": args.feedback_ratio,
        "device": args.device,
        # Dynamic loss weights
        "lambda_con_init": args.lambda_con_init,
        "lambda_proto_init": args.lambda_proto_init,
        "lambda_reg_init": args.lambda_reg_init,
        "lambda_con_decay_rate": args.lambda_con_decay_rate,
        "lambda_proto_increase_factor": args.lambda_proto_increase_factor,
        # Learning rate scheduler
        "scheduler_type": args.scheduler_type,
        "step_size": args.step_size,
        "gamma": args.gamma,
        "cosine_t_max": args.cosine_t_max,
    }

    if args.mode == "single":
        rich_interface.print_info(
            f"Training single category: {args.category}", "Mode: Single Category"
        )
        learner, scores = train_single_category(
            root_dir=args.root_dir,
            category=args.category,
            save_dir=args.save_dir,
            **training_kwargs,
        )

    elif args.mode == "all":
        rich_interface.print_info("Training all categories", "Mode: Multi-Category")
        train_all_categories(
            root_dir=args.root_dir,
            categories=args.categories,
            save_dir=args.save_dir,
            **training_kwargs,
        )

    elif args.mode == "eval":
        if args.model_path is None:
            rich_interface.print_error("model_path is required for evaluation mode")
            raise ValueError("model_path is required for evaluation mode")

        rich_interface.print_info(
            f"Evaluating model: {args.model_path}", "Mode: Evaluation"
        )
        evaluate_category(
            model_path=args.model_path,
            root_dir=args.root_dir,
            category=args.category,
            img_size=args.img_size,
            batch_size=args.batch_size,
        )

    # # Example dataset info
    # logger.info("\nDataset Information:")
    # multi_mvtec = MultiCategoryMVTecAD(args.root_dir)
    # multi_mvtec.print_dataset_info()


if __name__ == "__main__":
    main()

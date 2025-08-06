import time
from pathlib import Path
from typing import Any, Dict, List

from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

# Initialize rich console
console = Console()


class RichTrainingInterface:
    """Rich-enhanced interface for training visualization"""

    def __init__(self):
        self.console = console
        self.current_metrics = {}
        self.training_start_time = None

    def print_banner(self, title: str, subtitle: str = ""):
        """Print a beautiful banner"""
        banner_text = f"[bold blue]{title}[/bold blue]"
        if subtitle:
            banner_text += f"\n[dim]{subtitle}[/dim]"

        panel = Panel(
            Align.center(banner_text),
            border_style="blue",
            padding=(1, 2),
        )
        self.console.print(panel)
        self.console.print()

    def print_dataset_info(self, info: Dict[str, int]):
        """Print dataset information in a beautiful table"""
        table = Table(
            title="📊 Dataset Information",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Dataset Split", style="cyan", no_wrap=True)
        table.add_column("Samples", justify="right", style="green")
        table.add_column("Batches", justify="right", style="yellow")

        for split_name, samples in info.items():
            # Calculate batches assuming batch_size of 32 (you can pass this as parameter)
            batches = (samples + 31) // 32  # Ceiling division
            table.add_row(split_name, str(samples), str(batches))

        self.console.print(table)
        self.console.print()

    def start_training_session(
        self, category: str, num_epochs: int, hyperparams: Dict[str, Any]
    ):
        """Start a new training session with rich display"""
        self.training_start_time = time.time()

        # Print training info
        self.print_banner(f"🚀 Training: {category}", "Hierarchical Prototype Learning")

        # Hyperparameters table
        table = Table(
            title="⚙️ Hyperparameters", show_header=True, header_style="bold cyan"
        )
        table.add_column("Parameter", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        key_params = [
            ("Epochs", num_epochs),
            ("Learning Rate", hyperparams.get("learning_rate", "N/A")),
            ("Batch Size", hyperparams.get("batch_size", "N/A")),
            ("Feature Dim", hyperparams.get("d_model", "N/A")),
            ("K Coarse", hyperparams.get("k_coarse", "N/A")),
            ("K Fine", hyperparams.get("k_fine", "N/A")),
            ("Device", hyperparams.get("device", "N/A")),
        ]

        for param, value in key_params:
            table.add_row(param, str(value))

        self.console.print(table)
        self.console.print()

    def create_epoch_progress(self, num_epochs: int) -> Progress:
        """Create a progress bar for epochs"""
        return Progress(
            TextColumn("[bold blue]Epoch"),
            BarColumn(bar_width=30),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
        )

    def create_batch_progress(self, num_batches: int) -> Progress:
        """Create a progress bar for batches within an epoch"""
        return Progress(
            TextColumn("[bold green]Batch"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TextColumn("[cyan]Loss: [white]{task.fields[loss]:.4f}"),
            console=self.console,
        )

    def update_metrics_table(self, epoch: int, metrics: Dict[str, float]) -> Table:
        """Create/update metrics table"""
        table = Table(
            title=f"📈 Training Metrics - Epoch {epoch}",
            show_header=True,
            header_style="bold green",
        )
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", justify="right", style="white")
        table.add_column("Trend", justify="center", style="yellow")

        for metric_name, value in metrics.items():
            # Simple trend indicator (you can enhance this)
            trend = (
                "📈"
                if metric_name not in self.current_metrics
                else (
                    "📈"
                    if value > self.current_metrics[metric_name]
                    else "📉"
                    if value < self.current_metrics[metric_name]
                    else "➖"
                )
            )

            if "loss" in metric_name.lower():
                trend = (
                    "📉"
                    if metric_name not in self.current_metrics
                    else (
                        "📉"
                        if value < self.current_metrics[metric_name]
                        else "📈"
                        if value > self.current_metrics[metric_name]
                        else "➖"
                    )
                )

            display_name = metric_name.replace("_", " ").title()
            table.add_row(display_name, f"{value:.6f}", trend)

        self.current_metrics.update(metrics)
        return table

    def create_prototype_stats_table(self, stats: Dict[str, int]) -> Table:
        """Create prototype statistics table"""
        table = Table(
            title="🧠 Prototype Memory", show_header=True, header_style="bold purple"
        )
        table.add_column("Prototype Type", style="cyan", no_wrap=True)
        table.add_column("Count", justify="right", style="magenta")

        table.add_row("Coarse Prototypes", str(stats.get("num_coarse", 0)))
        table.add_row("Fine Prototypes", str(stats.get("num_fine_total", 0)))
        if stats.get("num_coarse", 0) > 0:
            avg_fine = stats.get("num_fine_total", 0) / stats.get("num_coarse", 1)
            table.add_row("Avg Fine/Coarse", f"{avg_fine:.2f}")

        return table

    def print_training_summary(
        self, category: str, final_metrics: Dict[str, float], model_path: str
    ):
        """Print training completion summary"""
        elapsed_time = (
            time.time() - self.training_start_time if self.training_start_time else 0
        )

        # Create summary panel
        summary_text = f"""[bold green]✅ Training Completed![/bold green]
[cyan]Category:[/cyan] {category}
[cyan]Training Time:[/cyan] {elapsed_time / 60:.1f} minutes
[cyan]Final Loss:[/cyan] {final_metrics.get("total_loss", "N/A"):.4f}
[cyan]Model Saved:[/cyan] {Path(model_path).name}
"""

        panel = Panel(
            summary_text,
            title="🎉 Success",
            border_style="green",
            padding=(1, 2),
        )
        self.console.print(panel)

    def print_evaluation_results(self, results: Dict[str, float]):
        """Print evaluation results in a beautiful format"""
        table = Table(
            title="📊 Evaluation Results", show_header=True, header_style="bold blue"
        )
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Score", justify="right", style="green")
        table.add_column("Performance", justify="center", style="yellow")

        # Add performance indicators
        for metric, score in results.items():
            if metric in ["auroc", "auprc"]:
                if score >= 0.95:
                    performance = "🌟 Excellent"
                elif score >= 0.90:
                    performance = "🔥 Great"
                elif score >= 0.80:
                    performance = "👍 Good"
                elif score >= 0.70:
                    performance = "⚠️ Fair"
                else:
                    performance = "❌ Poor"

                display_name = metric.upper()
                table.add_row(display_name, f"{score:.4f}", performance)
            elif metric.startswith("num_"):
                display_name = metric.replace("num_", "").replace("_", " ").title()
                table.add_row(display_name, str(int(score)), "")

        self.console.print(table)

    def create_multi_category_progress(self, categories: List[str]) -> Progress:
        """Create progress bar for multi-category training"""
        return Progress(
            TextColumn("[bold magenta]Categories"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TextColumn("[cyan]Current: [white]{task.fields[current]}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
        )

    def print_multi_category_summary(self, results: Dict[str, Dict]):
        """Print summary for multi-category training"""
        table = Table(
            title="🏆 Multi-Category Training Results",
            show_header=True,
            header_style="bold gold1",
        )
        table.add_column("Category", style="cyan", no_wrap=True)
        table.add_column("Status", justify="center", style="white")
        table.add_column("AUROC", justify="right", style="green")
        table.add_column("AUPRC", justify="right", style="blue")

        successful_count = 0
        for category, result in results.items():
            if result["status"] == "success":
                successful_count += 1
                status = "✅ Success"
                # You might need to extract AUROC/AUPRC from test_scores or add evaluation
                auroc = "N/A"  # Would need evaluation results
                auprc = "N/A"
            else:
                status = "❌ Failed"
                auroc = "N/A"
                auprc = "N/A"

            table.add_row(category, status, auroc, auprc)

        self.console.print(table)

        # Summary statistics
        total_categories = len(results)
        success_rate = (successful_count / total_categories) * 100

        summary_text = f"""[bold]Summary Statistics:[/bold]
[green]Successful:[/green] {successful_count}/{total_categories} ({success_rate:.1f}%)
[red]Failed:[/red] {total_categories - successful_count}/{total_categories}
"""

        panel = Panel(summary_text, title="📈 Overall Results", border_style="blue")
        self.console.print(panel)

    def print_error(self, message: str, category: str = ""):
        """Print error message with rich formatting"""
        error_text = "[bold red]❌ Error"
        if category:
            error_text += f" ({category})"
        error_text += f":[/bold red]\n{message}"

        panel = Panel(error_text, border_style="red", padding=(1, 2))
        self.console.print(panel)

    def print_warning(self, message: str):
        """Print warning message with rich formatting"""
        warning_text = f"[bold yellow]⚠️ Warning:[/bold yellow] {message}"
        self.console.print(warning_text)

    def print_info(self, message: str, title: str = ""):
        """Print info message with rich formatting"""
        if title:
            panel = Panel(
                f"[cyan]{message}[/cyan]", title=f"ℹ️ {title}", border_style="cyan"
            )
            self.console.print(panel)
        else:
            self.console.print(f"[cyan]ℹ️ {message}[/cyan]")

    def log_batch_loss(
        self,
        epoch: int,
        batch_idx: int,
        total_batches: int,
        losses: Dict[str, float],
        lambda_weights: Dict[str, float],
    ):
        """Log batch loss with rich formatting - called every N batches"""
        # Create a compact, informative loss display
        loss_components = []
        for loss_name, loss_value in losses.items():
            if loss_name != "total_loss":
                lambda_key = f"lambda_{loss_name.split('_')[0]}"
                lambda_val = lambda_weights.get(lambda_key, 1.0)
                loss_components.append(
                    f"[cyan]{loss_name}[/cyan]=[white]{loss_value:.4f}[/white]([yellow]λ={lambda_val:.3f}[/yellow])"
                )

        components_str = " + ".join(loss_components)

        progress_bar = "█" * int((batch_idx / total_batches) * 20)
        progress_bar = progress_bar.ljust(20, "░")

        self.console.print(
            f"[bold blue]Epoch {epoch}[/bold blue] [{progress_bar}] "
            f"[green]Batch {batch_idx:4d}/{total_batches}[/green] | "
            f"[bold red]Total Loss: {losses.get('total_loss', 0.0):.4f}[/bold red] | "
            f"{components_str}"
        )

    def log_epoch_summary(
        self,
        epoch: int,
        epoch_losses: Dict[str, float],
        learning_rate: float,
        elapsed_time: float,
    ):
        """Log epoch summary with rich formatting"""
        # Create a summary line with key metrics
        loss_summary = " | ".join(
            [
                f"[cyan]{name.replace('_', ' ').title()}[/cyan]: [white]{value:.4f}[/white]"
                for name, value in epoch_losses.items()
            ]
        )

        # self.console.print(
        #     f"\n[bold green]✅ Epoch {epoch} Complete[/bold green] "
        #     f"([yellow]{elapsed_time:.1f}s[/yellow]) | "
        #     f"[magenta]LR: {learning_rate:.6f}[/magenta]\n"
        #     f"   {loss_summary}\n"
        # )

    # def create_live_metrics_display(self) -> Table:
    #     """Create a live updating metrics table for training"""
    #     table = Table(
    #         title="🔄 Live Training Metrics",
    #         show_header=True,
    #         header_style="bold green",
    #         title_style="bold cyan",
    #     )
    #     table.add_column("Metric", style="cyan", no_wrap=True, width=15)
    #     table.add_column("Current", justify="right", style="white", width=10)
    #     table.add_column("Best", justify="right", style="green", width=10)
    #     table.add_column("Trend", justify="center", style="yellow", width=8)
    #     table.add_column("Weight (λ)", justify="right", style="magenta", width=12)
    #     return table

    # def update_live_metrics(
    #     self,
    #     table: Table,
    #     current_metrics: Dict[str, float],
    #     lambda_weights: Dict[str, float],
    #     best_metrics: Dict[str, float],
    # ):
    #     """Update live metrics table with current values"""
    #     # Clear existing rows
    #     table.rows.clear()

    #     for metric_name, current_value in current_metrics.items():
    #         best_value = best_metrics.get(metric_name, current_value)

    #         # Determine trend
    #         if metric_name not in self.current_metrics:
    #             trend = "🆕"
    #         else:
    #             prev_value = self.current_metrics[metric_name]
    #             if "loss" in metric_name.lower():
    #                 # For losses, lower is better
    #                 trend = (
    #                     "📉"
    #                     if current_value < prev_value
    #                     else "📈"
    #                     if current_value > prev_value
    #                     else "➖"
    #                 )
    #             else:
    #                 # For other metrics, higher might be better
    #                 trend = (
    #                     "📈"
    #                     if current_value > prev_value
    #                     else "📉"
    #                     if current_value < prev_value
    #                     else "➖"
    #                 )

    #         # Get lambda weight if applicable
    #         lambda_key = f"lambda_{metric_name.split('_')[0]}"
    #         lambda_val = lambda_weights.get(lambda_key, "N/A")
    #         lambda_str = (
    #             f"{lambda_val:.3f}"
    #             if isinstance(lambda_val, float)
    #             else str(lambda_val)
    #         )

    #         display_name = metric_name.replace("_", " ").title()
    #         table.add_row(
    #             display_name,
    #             f"{current_value:.4f}",
    #             f"{best_value:.4f}",
    #             trend,
    #             lambda_str,
    #         )

    #     # Update stored metrics
    #     self.current_metrics.update(current_metrics)


# Global instance for easy access
rich_interface = RichTrainingInterface()

import logging

import torch
from sklearn.cluster import KMeans

from ..logging import get_logger

logger = get_logger("Hierarchical Prototype Memory", logging.DEBUG)


class HierarchicalPrototypeMemory:
    """
    Hierarchical prototype memory with coarse and fine prototypes

    This class implements a two-level hierarchical prototype memory for anomaly detection:
    1. Coarse prototypes: High-level clusters representing broad normal patterns
    2. Fine prototypes: Detailed sub-clusters within each coarse cluster

    The hierarchy allows for:
    - Efficient similarity search
    - Adaptive prototype splitting based on anomaly feedback
    - Multi-scale pattern representation
    """

    def __init__(self, feature_dim: int, k_coarse: int, k_fine: int):
        """
        Initialize hierarchical prototype memory

        Args:
            feature_dim: Dimension of feature vectors
            k_coarse: Number of coarse prototypes
            k_fine: Maximum number of fine prototypes per coarse cluster
        """
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if k_coarse <= 0:
            raise ValueError("k_coarse must be positive")
        if k_fine <= 0:
            raise ValueError("k_fine must be positive")

        self.feature_dim = feature_dim
        self.k_coarse = k_coarse
        self.k_fine = k_fine

        # Coarse prototypes: (k_coarse, feature_dim)
        self.coarse_prototypes = None

        # Fine prototypes: dict mapping coarse_idx -> (k_fine_actual, feature_dim)
        self.fine_prototypes = {}

        # Prototype assignments for samples
        self.coarse_assignments = None

        # Statistics for monitoring
        self.total_splits = 0

    def initialize_prototypes(self, features: torch.Tensor):
        """
        Initialize prototypes using clustering

        Args:
            features: Input features tensor of shape (N, feature_dim)
        """
        if features.dim() != 2:
            raise ValueError(f"Expected 2D features tensor, got {features.dim()}D")
        if features.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected feature_dim={self.feature_dim}, got {features.shape[1]}"
            )
        if features.shape[0] < self.k_coarse:
            raise ValueError(
                f"Need at least {self.k_coarse} samples for {self.k_coarse} coarse prototypes"
            )

        features_np = features.detach().cpu().numpy()
        n_samples = features_np.shape[0]

        # Coarse clustering
        k_coarse_actual = min(self.k_coarse, n_samples)
        kmeans_coarse = KMeans(n_clusters=k_coarse_actual, random_state=42, n_init=10)
        coarse_labels = kmeans_coarse.fit_predict(features_np)
        self.coarse_prototypes = torch.tensor(
            kmeans_coarse.cluster_centers_, dtype=torch.float32, device=features.device
        )
        self.coarse_assignments = torch.tensor(coarse_labels, device=features.device)

        # Fine clustering for each coarse cluster
        total_fine_prototypes = 0
        for i in range(k_coarse_actual):
            cluster_mask = self.coarse_assignments == i
            cluster_size = cluster_mask.sum().item()

            if cluster_size > 0:
                cluster_features = features[cluster_mask].detach().cpu().numpy()

                # Handle case where cluster has fewer samples than k_fine
                k_fine_actual = min(self.k_fine, cluster_size)

                if k_fine_actual > 1:
                    kmeans_fine = KMeans(
                        n_clusters=int(k_fine_actual), random_state=42, n_init=10
                    )
                    kmeans_fine.fit(cluster_features)  # Just fit, don't need labels
                    self.fine_prototypes[i] = torch.tensor(
                        kmeans_fine.cluster_centers_,
                        dtype=torch.float32,
                        device=features.device,
                    )
                else:
                    # If only one sample or less, use the sample itself as prototype
                    self.fine_prototypes[i] = torch.tensor(
                        cluster_features[:1],
                        dtype=torch.float32,
                        device=features.device,
                    )

                total_fine_prototypes += self.fine_prototypes[i].shape[0]

        logger.info(
            "Initialized %d coarse prototypes and %d total fine prototypes",
            k_coarse_actual,
            total_fine_prototypes,
        )

    def compute_anomaly_score(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute anomaly scores using hierarchical prototypes

        Args:
            features: Input features tensor of shape (N, feature_dim)

        Returns:
            Anomaly scores tensor of shape (N,) where higher values indicate more anomalous samples
        """
        if self.coarse_prototypes is None:
            raise ValueError(
                "Prototypes not initialized. Call initialize_prototypes() first."
            )

        if features.dim() != 2:
            raise ValueError(f"Expected 2D features tensor, got {features.dim()}D")
        if features.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected feature_dim={self.feature_dim}, got {features.shape[1]}"
            )

        return self._compute_scores_vectorized(features)

    def _compute_scores_vectorized(self, features: torch.Tensor) -> torch.Tensor:
        """Vectorized implementation for better performance"""
        assert self.coarse_prototypes is not None, "Prototypes must be initialized"

        batch_size = features.shape[0]
        scores = torch.zeros(batch_size, device=features.device)

        # Find closest coarse prototypes for all features at once
        coarse_distances = torch.cdist(
            features, self.coarse_prototypes
        )  # (B, K_coarse)
        closest_coarse_indices = torch.argmin(coarse_distances, dim=1)  # (B,)

        # Process each unique coarse cluster
        unique_coarse_indices = torch.unique(closest_coarse_indices)

        for coarse_idx in unique_coarse_indices:
            coarse_idx_int = coarse_idx.item()

            # Get mask for features assigned to this coarse prototype
            mask = closest_coarse_indices == coarse_idx
            batch_features = features[mask]

            if coarse_idx_int in self.fine_prototypes:
                fine_prototypes = self.fine_prototypes[coarse_idx_int]
                fine_distances = torch.cdist(batch_features, fine_prototypes)
                min_distances = torch.min(fine_distances, dim=1)[0]
                scores[mask] = min_distances
            else:
                # Fallback to coarse distance
                scores[mask] = coarse_distances[mask, coarse_idx]

        return scores

    def _compute_scores_loop(self, features: torch.Tensor) -> torch.Tensor:
        """Original loop-based implementation for reference"""
        assert self.coarse_prototypes is not None, "Prototypes must be initialized"

        batch_size = features.shape[0]
        scores = torch.zeros(batch_size, device=features.device)

        for i in range(batch_size):
            feature = features[i : i + 1]  # (1, feature_dim)

            # Find closest coarse prototype
            coarse_distances = torch.cdist(
                feature, self.coarse_prototypes
            )  # (1, k_coarse)
            closest_coarse_idx = torch.argmin(coarse_distances, dim=1).item()

            # Compute distance to fine prototypes in the closest coarse cluster
            if closest_coarse_idx in self.fine_prototypes:
                fine_prototypes = self.fine_prototypes[closest_coarse_idx]
                fine_distances = torch.cdist(
                    feature, fine_prototypes
                )  # (1, k_fine_actual)
                min_fine_distance = torch.min(fine_distances)
                scores[i] = min_fine_distance
            else:
                # Fallback to coarse distance
                scores[i] = coarse_distances[0, int(closest_coarse_idx)]

        return scores

    def split_prototype(
        self, coarse_idx: int, fine_idx: int, anomaly_features: torch.Tensor
    ):
        """
        Split a fine prototype when anomaly feedback is received

        Args:
            coarse_idx: Index of the coarse prototype cluster
            fine_idx: Index of the fine prototype to split
            anomaly_features: Tensor of anomalous features that triggered the split
        """
        if coarse_idx not in self.fine_prototypes:
            logger.warning("Coarse index %d not found in fine prototypes", coarse_idx)
            return

        fine_prototypes = self.fine_prototypes[coarse_idx]
        if fine_idx >= len(fine_prototypes):
            logger.warning(
                "Fine index %d out of range for coarse cluster %d (max: %d)",
                fine_idx,
                coarse_idx,
                len(fine_prototypes) - 1,
            )
            return

        # Validate input
        if anomaly_features.dim() != 2:
            raise ValueError(
                f"Expected 2D anomaly_features, got {anomaly_features.dim()}D"
            )
        if anomaly_features.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected feature_dim={self.feature_dim}, got {anomaly_features.shape[1]}"
            )

        # Create new prototypes by splitting the affected one
        original_prototype = fine_prototypes[fine_idx : fine_idx + 1]

        # Combine original prototype with anomaly features for re-clustering
        combined_features = torch.cat([original_prototype, anomaly_features], dim=0)

        # Re-cluster into 2 new prototypes
        if len(combined_features) >= 2:
            combined_np = combined_features.detach().cpu().numpy()
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            kmeans.fit(combined_np)
            new_prototypes = torch.tensor(
                kmeans.cluster_centers_,
                dtype=torch.float32,
                device=anomaly_features.device,
            )

            # Replace the original prototype with new ones
            updated_prototypes = torch.cat(
                [
                    fine_prototypes[:fine_idx],
                    new_prototypes,
                    fine_prototypes[fine_idx + 1 :],
                ],
                dim=0,
            )

            self.fine_prototypes[coarse_idx] = updated_prototypes
            self.total_splits += 1

            logger.info(
                "Split prototype (%d, %d) into 2 new prototypes (total splits: %d)",
                coarse_idx,
                fine_idx,
                self.total_splits,
            )
        else:
            logger.warning(
                "Cannot split prototype (%d, %d): insufficient features (%d)",
                coarse_idx,
                fine_idx,
                len(combined_features),
            )

    def get_statistics(self) -> dict:
        """
        Get statistics about the current prototype memory state

        Returns:
            Dictionary containing statistics about prototypes and splits
        """
        if self.coarse_prototypes is None:
            return {"initialized": False}

        total_fine_prototypes = sum(
            prototypes.shape[0] for prototypes in self.fine_prototypes.values()
        )

        fine_counts = [
            prototypes.shape[0] for prototypes in self.fine_prototypes.values()
        ]

        return {
            "initialized": True,
            "num_coarse_prototypes": self.coarse_prototypes.shape[0],
            "num_fine_clusters": len(self.fine_prototypes),
            "total_fine_prototypes": total_fine_prototypes,
            "avg_fine_per_coarse": total_fine_prototypes / len(self.fine_prototypes)
            if self.fine_prototypes
            else 0,
            "min_fine_per_coarse": min(fine_counts) if fine_counts else 0,
            "max_fine_per_coarse": max(fine_counts) if fine_counts else 0,
            "total_splits": self.total_splits,
        }

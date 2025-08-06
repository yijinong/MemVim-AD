from typing import Union

import torch


class AnomalyScorer:
    def __init__(self, prototypes: torch.Tensor):
        """
        Initialize with learned prototypes

        Args:
            prototypes: Tensor of shape [K, C] where K = num_prototypes, C = feature_dim
        """
        self.prototypes = prototypes
        self._validate_prototypes()

    def _validate_prototypes(self):
        if not isinstance(self.prototypes, torch.Tensor):
            raise TypeError("Prototypes must be torch.Tensor")

        if len(self.prototypes.shape) != 2:
            raise ValueError("Prototypes should be 2D [K, C] tensor")

        if torch.any(torch.isnan(self.prototypes)):
            raise ValueError("Prototypes contain NaN values")

    def score(self, feats: Union[torch.Tensor, list]) -> torch.Tensor:
        """
        Compute anomaly scores for input features

        Args:
            feats: Input features [N, C] or list of feature vectors

        Returns:
            anomaly_scores: Tensor of shape [N] containing distances
        """
        if isinstance(feats, list):
            feats = torch.stack(feats)

        if len(feats.shape) == 1:
            feats = feats.unsqueeze(0)

        if feats.ndim == 1:
            feats = feats.unsqueeze(0)
        assert feats.ndim == 2, "Expected [N, C] feature shape"

        # Compute L2 distances to all prototypes
        dist = torch.cdist(feats, self.prototypes, p=2)

        anomaly_scores, _ = torch.min(dist, dim=1)

        return anomaly_scores

    def score_w_softmin(self, feats: torch.Tensor, temp: float = 0.1) -> torch.Tensor:
        """
        Alternating scoring using soft minimum (smooth approximation)

        Args:
            temp: Controls sharpness of softmin (lower = sharper)
        """
        dist = torch.cdist(feats, self.prototypes, p=2)
        weights = torch.softmax(-dist / temp, dim=1)

        return torch.sum(weights * dist, dim=1)


def main() -> None:
    prototypes = torch.randn(10, 128)
    scorer = AnomalyScorer(prototypes)
    test_feats = torch.randn(5, 128)

    scores = scorer.score(test_feats)
    print(f"Anomaly score: {scores}")

    soft_scores = scorer.score_w_softmin(test_feats)
    print(f"Softmin score: {soft_scores}")


if __name__ == "__main__":
    main()

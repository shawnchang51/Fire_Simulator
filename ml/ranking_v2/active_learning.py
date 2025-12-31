"""
Active Learning Strategies for Efficient Data Collection

Active learning allows the model to select which configurations to simulate,
reducing the total number of expensive simulations needed while maximizing
model performance.

Strategies:
1. Uncertainty Sampling: Select samples where model is most uncertain
2. Query-by-Committee: Select samples where ensemble members disagree
3. Expected Model Change: Select samples that would most change the model
4. Diversity Sampling: Select diverse samples to cover the space
5. Batch Mode Active Learning: Select batches with diversity + uncertainty
"""

from typing import Dict, List, Optional, Tuple, Callable, Union
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod
from collections import defaultdict

from .uncertainty import MCDropoutWrapper, UncertaintyEstimate, DeepEnsemble
from .model import CrossAttentionRanker
from .config import RankingV2Config


@dataclass
class ActiveLearningState:
    """State container for active learning loop."""
    labeled_indices: List[int]
    unlabeled_indices: List[int]
    query_history: List[List[int]]
    performance_history: List[Dict[str, float]]
    iteration: int


@dataclass
class QueryResult:
    """Result of a query selection."""
    indices: List[int]
    scores: np.ndarray
    strategy_info: Dict[str, any]


class AcquisitionFunction(ABC):
    """Base class for acquisition functions."""

    @abstractmethod
    def __call__(
        self,
        model: nn.Module,
        unlabeled_data: List[Dict],
        device: torch.device,
    ) -> np.ndarray:
        """
        Compute acquisition scores for unlabeled samples.

        Args:
            model: Current model
            unlabeled_data: List of unlabeled samples
            device: Device for computation

        Returns:
            Acquisition scores (higher = more valuable to label)
        """
        pass


class UncertaintySampling(AcquisitionFunction):
    """
    Uncertainty-based acquisition function.

    Selects samples where the model is most uncertain about the prediction.
    """

    def __init__(
        self,
        n_mc_samples: int = 30,
        uncertainty_type: str = 'entropy',
    ):
        """
        Initialize uncertainty sampling.

        Args:
            n_mc_samples: Number of MC dropout samples
            uncertainty_type: Type of uncertainty ('entropy', 'variance', 'margin')
        """
        self.n_mc_samples = n_mc_samples
        self.uncertainty_type = uncertainty_type

    def __call__(
        self,
        model: nn.Module,
        unlabeled_data: List[Dict],
        device: torch.device,
    ) -> np.ndarray:
        """Compute uncertainty scores for unlabeled pairs."""
        # Wrap model for MC dropout if needed
        if not isinstance(model, MCDropoutWrapper):
            mc_model = MCDropoutWrapper(model, n_samples=self.n_mc_samples)
        else:
            mc_model = model

        mc_model.to(device)
        mc_model.eval()

        scores = []

        for sample in unlabeled_data:
            grid_a = sample['grid_a'].unsqueeze(0).to(device)
            scenario_a = sample['scenario_a'].unsqueeze(0).to(device)
            grid_b = sample['grid_b'].unsqueeze(0).to(device)
            scenario_b = sample['scenario_b'].unsqueeze(0).to(device)

            uncertainty = mc_model.predict_with_uncertainty(
                grid_a, scenario_a, grid_b, scenario_b
            )

            if self.uncertainty_type == 'entropy':
                score = uncertainty.entropy.item()
            elif self.uncertainty_type == 'variance':
                score = uncertainty.std_logit.item()
            elif self.uncertainty_type == 'margin':
                # Margin = |P(A>B) - 0.5|, lower margin = more uncertain
                score = 1.0 - abs(uncertainty.mean_prob.item() - 0.5) * 2
            else:
                score = uncertainty.entropy.item()

            scores.append(score)

        return np.array(scores)


class QueryByCommittee(AcquisitionFunction):
    """
    Query by Committee acquisition function.

    Uses an ensemble of models and selects samples where
    committee members disagree most.
    """

    def __init__(self, disagreement_type: str = 'vote_entropy'):
        """
        Initialize QBC.

        Args:
            disagreement_type: Type of disagreement measure
                              ('vote_entropy', 'kl_divergence', 'variance')
        """
        self.disagreement_type = disagreement_type

    def __call__(
        self,
        model: nn.Module,
        unlabeled_data: List[Dict],
        device: torch.device,
    ) -> np.ndarray:
        """Compute disagreement scores for unlabeled pairs."""
        if not isinstance(model, DeepEnsemble):
            raise ValueError("QueryByCommittee requires a DeepEnsemble model")

        model.to(device)

        scores = []

        for sample in unlabeled_data:
            grid_a = sample['grid_a'].unsqueeze(0).to(device)
            scenario_a = sample['scenario_a'].unsqueeze(0).to(device)
            grid_b = sample['grid_b'].unsqueeze(0).to(device)
            scenario_b = sample['scenario_b'].unsqueeze(0).to(device)

            # Get predictions from all ensemble members
            predictions = []
            for idx, member in enumerate(model.models):
                if not model._trained[idx]:
                    continue
                member.eval()
                with torch.no_grad():
                    outputs = member(grid_a, scenario_a, grid_b, scenario_b)
                    prob = torch.sigmoid(outputs['logit']).item()
                    predictions.append(prob)

            predictions = np.array(predictions)

            if self.disagreement_type == 'vote_entropy':
                # Vote entropy based on binary predictions
                votes = (predictions > 0.5).mean()
                if votes == 0 or votes == 1:
                    score = 0.0
                else:
                    score = -votes * np.log(votes) - (1 - votes) * np.log(1 - votes)
            elif self.disagreement_type == 'variance':
                score = predictions.var()
            elif self.disagreement_type == 'kl_divergence':
                # Average KL from each member to consensus
                consensus = predictions.mean()
                kl = 0
                for p in predictions:
                    eps = 1e-8
                    kl += p * np.log((p + eps) / (consensus + eps)) + \
                          (1 - p) * np.log((1 - p + eps) / (1 - consensus + eps))
                score = kl / len(predictions)
            else:
                score = predictions.var()

            scores.append(score)

        return np.array(scores)


class ExpectedModelChange(AcquisitionFunction):
    """
    Expected Model Change (EMC) acquisition function.

    Selects samples that would cause the largest expected change
    in model parameters if labeled.
    """

    def __init__(self):
        pass

    def __call__(
        self,
        model: nn.Module,
        unlabeled_data: List[Dict],
        device: torch.device,
    ) -> np.ndarray:
        """Compute expected gradient length for unlabeled pairs."""
        model.to(device)
        model.train()  # Enable gradients

        scores = []

        for sample in unlabeled_data:
            grid_a = sample['grid_a'].unsqueeze(0).to(device)
            scenario_a = sample['scenario_a'].unsqueeze(0).to(device)
            grid_b = sample['grid_b'].unsqueeze(0).to(device)
            scenario_b = sample['scenario_b'].unsqueeze(0).to(device)

            # Compute gradients for both possible labels
            gradient_norms = []

            for label in [0, 1]:
                model.zero_grad()
                outputs = model(grid_a, scenario_a, grid_b, scenario_b)
                logit = outputs['logit']

                # BCE loss with hypothetical label
                loss = nn.BCEWithLogitsLoss()(logit, torch.tensor([float(label)], device=device))
                loss.backward()

                # Compute gradient norm
                total_norm = 0.0
                for param in model.parameters():
                    if param.grad is not None:
                        total_norm += param.grad.norm(2).item() ** 2
                total_norm = np.sqrt(total_norm)
                gradient_norms.append(total_norm)

            # Expected gradient: average over possible labels weighted by model's prediction
            with torch.no_grad():
                outputs = model(grid_a, scenario_a, grid_b, scenario_b)
                prob = torch.sigmoid(outputs['logit']).item()

            expected_gradient = prob * gradient_norms[1] + (1 - prob) * gradient_norms[0]
            scores.append(expected_gradient)

        model.eval()
        return np.array(scores)


class DiversitySampling(AcquisitionFunction):
    """
    Diversity-based acquisition function.

    Selects samples that are diverse in the feature space,
    ensuring good coverage of the input distribution.
    """

    def __init__(
        self,
        n_clusters: int = 10,
        use_latent: bool = True,
    ):
        """
        Initialize diversity sampling.

        Args:
            n_clusters: Number of clusters for k-means
            use_latent: Use latent representations (True) or raw features (False)
        """
        self.n_clusters = n_clusters
        self.use_latent = use_latent

    def __call__(
        self,
        model: nn.Module,
        unlabeled_data: List[Dict],
        device: torch.device,
    ) -> np.ndarray:
        """Compute diversity scores based on distance to cluster centers."""
        from sklearn.cluster import KMeans

        model.to(device)
        model.eval()

        # Extract features
        features = []
        with torch.no_grad():
            for sample in unlabeled_data:
                grid_a = sample['grid_a'].unsqueeze(0).to(device)
                grid_b = sample['grid_b'].unsqueeze(0).to(device)

                if self.use_latent and hasattr(model, 'encoder'):
                    latent_a = model.encoder(grid_a)
                    latent_b = model.encoder(grid_b)
                    feat = torch.cat([latent_a, latent_b], dim=1).cpu().numpy()
                else:
                    # Use raw grid statistics
                    feat = torch.cat([
                        grid_a.mean(dim=(2, 3)),
                        grid_b.mean(dim=(2, 3)),
                    ], dim=1).cpu().numpy()

                features.append(feat.squeeze())

        features = np.array(features)

        # Cluster features
        n_clusters = min(self.n_clusters, len(features))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(features)

        # Compute distance to nearest cluster center
        # Higher distance = more diverse / underrepresented
        distances = kmeans.transform(features).min(axis=1)

        return distances


class BatchModeSampler:
    """
    Batch mode active learning sampler.

    Selects a batch of samples that are both informative and diverse.
    """

    def __init__(
        self,
        acquisition_fn: AcquisitionFunction,
        batch_size: int = 32,
        diversity_weight: float = 0.3,
    ):
        """
        Initialize batch mode sampler.

        Args:
            acquisition_fn: Base acquisition function
            batch_size: Number of samples to select per batch
            diversity_weight: Weight for diversity vs acquisition score
        """
        self.acquisition_fn = acquisition_fn
        self.batch_size = batch_size
        self.diversity_weight = diversity_weight

    def select_batch(
        self,
        model: nn.Module,
        unlabeled_data: List[Dict],
        device: torch.device,
    ) -> QueryResult:
        """
        Select a diverse batch of informative samples.

        Uses a greedy algorithm that balances acquisition score
        with diversity from already selected samples.
        """
        # Compute acquisition scores
        acquisition_scores = self.acquisition_fn(model, unlabeled_data, device)

        # Extract features for diversity computation
        model.eval()
        features = []
        with torch.no_grad():
            for sample in unlabeled_data:
                grid_a = sample['grid_a'].unsqueeze(0).to(device)
                if hasattr(model, 'encoder'):
                    feat = model.encoder(grid_a).cpu().numpy().squeeze()
                else:
                    feat = grid_a.mean(dim=(2, 3)).cpu().numpy().squeeze()
                features.append(feat)
        features = np.array(features)

        # Greedy selection with diversity
        selected_indices = []
        remaining_indices = list(range(len(unlabeled_data)))

        for _ in range(min(self.batch_size, len(unlabeled_data))):
            if not remaining_indices:
                break

            # Compute combined scores
            combined_scores = []
            for idx in remaining_indices:
                acq_score = acquisition_scores[idx]

                # Diversity score: minimum distance to already selected
                if selected_indices:
                    selected_features = features[selected_indices]
                    distances = np.linalg.norm(
                        selected_features - features[idx], axis=1
                    )
                    diversity_score = distances.min()
                else:
                    diversity_score = 1.0

                # Combine scores
                combined = (1 - self.diversity_weight) * acq_score + \
                          self.diversity_weight * diversity_score

                combined_scores.append((idx, combined))

            # Select best
            best_idx = max(combined_scores, key=lambda x: x[1])[0]
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        return QueryResult(
            indices=selected_indices,
            scores=acquisition_scores[selected_indices],
            strategy_info={
                'batch_size': len(selected_indices),
                'diversity_weight': self.diversity_weight,
            },
        )


class ActiveLearningLoop:
    """
    Main active learning loop manager.

    Orchestrates the iterative process of:
    1. Training model on labeled data
    2. Selecting unlabeled samples to query
    3. Obtaining labels (via simulation)
    4. Adding to labeled set and repeating
    """

    def __init__(
        self,
        model: nn.Module,
        acquisition_fn: AcquisitionFunction,
        config: RankingV2Config,
        query_batch_size: int = 32,
        initial_labeled_size: int = 100,
        max_queries: int = 1000,
        device: torch.device = torch.device('cpu'),
    ):
        """
        Initialize active learning loop.

        Args:
            model: Ranking model
            acquisition_fn: Acquisition function for sample selection
            config: Model configuration
            query_batch_size: Number of samples to query per iteration
            initial_labeled_size: Size of initial labeled set
            max_queries: Maximum total queries allowed
            device: Device for computation
        """
        self.model = model.to(device)
        self.acquisition_fn = acquisition_fn
        self.config = config
        self.query_batch_size = query_batch_size
        self.initial_labeled_size = initial_labeled_size
        self.max_queries = max_queries
        self.device = device

        # State
        self.state = ActiveLearningState(
            labeled_indices=[],
            unlabeled_indices=[],
            query_history=[],
            performance_history=[],
            iteration=0,
        )

        # Batch sampler for diversity
        self.batch_sampler = BatchModeSampler(
            acquisition_fn,
            batch_size=query_batch_size,
        )

    def initialize(
        self,
        all_data: List[Dict],
        initial_indices: Optional[List[int]] = None,
    ):
        """
        Initialize the active learning state.

        Args:
            all_data: Complete dataset
            initial_indices: Optional initial labeled indices (random if None)
        """
        n_samples = len(all_data)

        if initial_indices is None:
            # Random initial selection
            initial_indices = np.random.choice(
                n_samples,
                size=min(self.initial_labeled_size, n_samples),
                replace=False,
            ).tolist()

        self.state.labeled_indices = initial_indices
        self.state.unlabeled_indices = [
            i for i in range(n_samples) if i not in initial_indices
        ]
        self.state.iteration = 0
        self.state.query_history = [initial_indices]

    def query(
        self,
        unlabeled_data: List[Dict],
    ) -> QueryResult:
        """
        Select samples to query for labels.

        Args:
            unlabeled_data: List of unlabeled samples

        Returns:
            QueryResult with selected indices
        """
        return self.batch_sampler.select_batch(
            self.model,
            unlabeled_data,
            self.device,
        )

    def update(
        self,
        query_result: QueryResult,
        new_labels: List[float],
    ):
        """
        Update state with newly labeled samples.

        Args:
            query_result: Result from query()
            new_labels: Labels for queried samples
        """
        # Map query indices to global indices
        global_indices = [
            self.state.unlabeled_indices[i] for i in query_result.indices
        ]

        # Update labeled/unlabeled sets
        self.state.labeled_indices.extend(global_indices)
        for idx in sorted(query_result.indices, reverse=True):
            self.state.unlabeled_indices.pop(idx)

        self.state.query_history.append(global_indices)
        self.state.iteration += 1

    def get_labeled_data(
        self,
        all_data: List[Dict],
    ) -> List[Dict]:
        """Get currently labeled data."""
        return [all_data[i] for i in self.state.labeled_indices]

    def get_unlabeled_data(
        self,
        all_data: List[Dict],
    ) -> List[Dict]:
        """Get currently unlabeled data."""
        return [all_data[i] for i in self.state.unlabeled_indices]

    @property
    def total_queries(self) -> int:
        """Total number of samples queried so far."""
        return len(self.state.labeled_indices)

    @property
    def budget_remaining(self) -> int:
        """Remaining query budget."""
        return max(0, self.max_queries - self.total_queries)

    def should_stop(self) -> bool:
        """Check if active learning should stop."""
        return (
            self.total_queries >= self.max_queries or
            len(self.state.unlabeled_indices) == 0
        )


class SimulationOracle:
    """
    Oracle that provides labels via simulation.

    Interface for connecting active learning to the fire simulator.
    """

    def __init__(
        self,
        simulator_fn: Callable,
        cache_results: bool = True,
    ):
        """
        Initialize simulation oracle.

        Args:
            simulator_fn: Function that takes a configuration and returns metrics
            cache_results: Whether to cache simulation results
        """
        self.simulator_fn = simulator_fn
        self.cache_results = cache_results
        self._cache = {}

    def get_label(
        self,
        config_a: Dict,
        config_b: Dict,
    ) -> Tuple[float, Dict]:
        """
        Get ranking label for a configuration pair.

        Args:
            config_a: First configuration
            config_b: Second configuration

        Returns:
            (label, metrics) where label = 1 if A > B, 0 otherwise
        """
        # Create cache keys
        key_a = self._make_key(config_a)
        key_b = self._make_key(config_b)

        # Get metrics for each config
        if self.cache_results and key_a in self._cache:
            metrics_a = self._cache[key_a]
        else:
            metrics_a = self.simulator_fn(config_a)
            if self.cache_results:
                self._cache[key_a] = metrics_a

        if self.cache_results and key_b in self._cache:
            metrics_b = self._cache[key_b]
        else:
            metrics_b = self.simulator_fn(config_b)
            if self.cache_results:
                self._cache[key_b] = metrics_b

        # Compute label based on survival rate (or other metric)
        score_a = metrics_a.get('survival_rate', metrics_a.get('score', 0))
        score_b = metrics_b.get('survival_rate', metrics_b.get('score', 0))

        label = 1.0 if score_a > score_b else 0.0

        combined_metrics = {
            'score_a': score_a,
            'score_b': score_b,
            'metrics_a': metrics_a,
            'metrics_b': metrics_b,
        }

        return label, combined_metrics

    def _make_key(self, config: Dict) -> str:
        """Create hashable key for configuration."""
        import json
        return json.dumps(config, sort_keys=True)

    def get_batch_labels(
        self,
        config_pairs: List[Tuple[Dict, Dict]],
    ) -> List[Tuple[float, Dict]]:
        """Get labels for a batch of configuration pairs."""
        return [self.get_label(a, b) for a, b in config_pairs]


def create_acquisition_function(
    strategy: str = 'uncertainty',
    **kwargs,
) -> AcquisitionFunction:
    """
    Factory function to create acquisition function.

    Args:
        strategy: Strategy name ('uncertainty', 'qbc', 'emc', 'diversity')
        **kwargs: Strategy-specific arguments

    Returns:
        AcquisitionFunction instance
    """
    if strategy == 'uncertainty':
        return UncertaintySampling(
            n_mc_samples=kwargs.get('n_mc_samples', 30),
            uncertainty_type=kwargs.get('uncertainty_type', 'entropy'),
        )
    elif strategy == 'qbc':
        return QueryByCommittee(
            disagreement_type=kwargs.get('disagreement_type', 'vote_entropy'),
        )
    elif strategy == 'emc':
        return ExpectedModelChange()
    elif strategy == 'diversity':
        return DiversitySampling(
            n_clusters=kwargs.get('n_clusters', 10),
            use_latent=kwargs.get('use_latent', True),
        )
    else:
        raise ValueError(f"Unknown acquisition strategy: {strategy}")


def compute_learning_curve(
    performance_history: List[Dict[str, float]],
    metric: str = 'accuracy',
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute learning curve from performance history.

    Args:
        performance_history: List of performance dicts
        metric: Metric to plot

    Returns:
        (n_samples, metric_values) arrays
    """
    n_samples = np.arange(1, len(performance_history) + 1)
    metric_values = np.array([h.get(metric, 0) for h in performance_history])

    return n_samples, metric_values

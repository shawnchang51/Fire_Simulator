"""
Continual Learning for Online Model Updates

Enables the ranking model to continuously learn from new data
without forgetting previously learned knowledge.

Approaches:
1. Experience Replay: Store and replay old samples
2. Elastic Weight Consolidation (EWC): Protect important parameters
3. Progressive Networks: Add new capacity for new tasks
4. Knowledge Distillation: Transfer knowledge from old to new model
"""

from typing import Dict, List, Optional, Tuple, Union, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from collections import deque
import random
from copy import deepcopy

from .model import CrossAttentionRanker, FloorPlanEncoder
from .config import RankingV2Config


@dataclass
class Experience:
    """Single experience for replay buffer."""
    grid_a: torch.Tensor
    scenario_a: torch.Tensor
    grid_b: torch.Tensor
    scenario_b: torch.Tensor
    label: torch.Tensor
    task_id: int = 0
    importance: float = 1.0


@dataclass
class ContinualLearningState:
    """State of continual learning process."""
    current_task: int
    tasks_seen: List[int]
    total_samples_seen: int
    forgetting_measure: Dict[int, float]  # Per-task accuracy degradation
    forward_transfer: Dict[int, float]  # Transfer to new tasks


class ExperienceReplayBuffer:
    """
    Experience replay buffer for continual learning.

    Stores past experiences and samples them during training
    to prevent catastrophic forgetting.
    """

    def __init__(
        self,
        capacity: int = 10000,
        sample_strategy: str = "uniform",  # "uniform", "reservoir", "priority"
    ):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum buffer size
            sample_strategy: Sampling strategy
        """
        self.capacity = capacity
        self.sample_strategy = sample_strategy
        self.buffer: deque = deque(maxlen=capacity)
        self.priorities: deque = deque(maxlen=capacity)

    def add(
        self,
        experience: Experience,
        priority: float = 1.0,
    ):
        """
        Add experience to buffer.

        Args:
            experience: Experience to add
            priority: Priority for prioritized sampling
        """
        self.buffer.append(experience)
        self.priorities.append(priority)

    def add_batch(
        self,
        grid_a: torch.Tensor,
        scenario_a: torch.Tensor,
        grid_b: torch.Tensor,
        scenario_b: torch.Tensor,
        labels: torch.Tensor,
        task_id: int = 0,
    ):
        """Add batch of experiences."""
        batch_size = grid_a.size(0)
        for i in range(batch_size):
            exp = Experience(
                grid_a=grid_a[i].cpu(),
                scenario_a=scenario_a[i].cpu(),
                grid_b=grid_b[i].cpu(),
                scenario_b=scenario_b[i].cpu(),
                label=labels[i].cpu(),
                task_id=task_id,
            )
            self.add(exp)

    def sample(self, batch_size: int) -> List[Experience]:
        """
        Sample batch of experiences.

        Args:
            batch_size: Number of samples

        Returns:
            List of sampled experiences
        """
        if len(self.buffer) < batch_size:
            return list(self.buffer)

        if self.sample_strategy == "uniform":
            return random.sample(list(self.buffer), batch_size)

        elif self.sample_strategy == "priority":
            priorities = np.array(list(self.priorities))
            probs = priorities / priorities.sum()
            indices = np.random.choice(
                len(self.buffer),
                size=batch_size,
                replace=False,
                p=probs,
            )
            return [self.buffer[i] for i in indices]

        elif self.sample_strategy == "reservoir":
            # Reservoir sampling (already handled by deque)
            return random.sample(list(self.buffer), batch_size)

        else:
            return random.sample(list(self.buffer), batch_size)

    def sample_per_task(
        self,
        batch_size: int,
        task_ids: Optional[List[int]] = None,
    ) -> Dict[int, List[Experience]]:
        """
        Sample experiences grouped by task.

        Args:
            batch_size: Samples per task
            task_ids: Which tasks to sample from (None = all)

        Returns:
            Dict mapping task_id to experiences
        """
        # Group by task
        task_experiences: Dict[int, List[Experience]] = {}
        for exp in self.buffer:
            tid = exp.task_id
            if task_ids is None or tid in task_ids:
                if tid not in task_experiences:
                    task_experiences[tid] = []
                task_experiences[tid].append(exp)

        # Sample from each task
        sampled = {}
        for tid, exps in task_experiences.items():
            n = min(batch_size, len(exps))
            sampled[tid] = random.sample(exps, n)

        return sampled

    def collate(
        self,
        experiences: List[Experience],
    ) -> Dict[str, torch.Tensor]:
        """Collate experiences into batch tensors."""
        return {
            'grid_a': torch.stack([e.grid_a for e in experiences]),
            'scenario_a': torch.stack([e.scenario_a for e in experiences]),
            'grid_b': torch.stack([e.grid_b for e in experiences]),
            'scenario_b': torch.stack([e.scenario_b for e in experiences]),
            'label': torch.stack([e.label for e in experiences]),
            'task_id': torch.tensor([e.task_id for e in experiences]),
        }

    def __len__(self) -> int:
        return len(self.buffer)


class EWC(nn.Module):
    """
    Elastic Weight Consolidation for continual learning.

    Protects important parameters by adding a regularization term
    that penalizes changes to parameters important for previous tasks.
    """

    def __init__(
        self,
        model: CrossAttentionRanker,
        ewc_lambda: float = 100.0,
    ):
        """
        Initialize EWC.

        Args:
            model: Base model
            ewc_lambda: EWC regularization strength
        """
        super().__init__()

        self.model = model
        self.ewc_lambda = ewc_lambda

        # Store Fisher information and optimal parameters per task
        self.fisher_matrices: Dict[int, Dict[str, torch.Tensor]] = {}
        self.optimal_params: Dict[int, Dict[str, torch.Tensor]] = {}

    def compute_fisher(
        self,
        dataloader: torch.utils.data.DataLoader,
        task_id: int,
        device: torch.device,
        n_samples: int = 1000,
    ):
        """
        Compute Fisher information matrix for current task.

        Args:
            dataloader: Task dataloader
            task_id: Task identifier
            device: Device
            n_samples: Number of samples for estimation
        """
        self.model.to(device)
        self.model.eval()

        # Initialize Fisher
        fisher = {}
        for name, param in self.model.named_parameters():
            fisher[name] = torch.zeros_like(param)

        n_computed = 0

        for batch in dataloader:
            if n_computed >= n_samples:
                break

            self.model.zero_grad()

            outputs = self.model(
                batch['grid_a'].to(device),
                batch['scenario_a'].to(device),
                batch['grid_b'].to(device),
                batch['scenario_b'].to(device),
            )

            # Use log-likelihood as objective
            log_prob = F.logsigmoid(outputs['logit'])
            log_prob.sum().backward()

            # Accumulate squared gradients
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.data ** 2

            n_computed += batch['grid_a'].size(0)

        # Normalize
        for name in fisher:
            fisher[name] /= n_computed

        self.fisher_matrices[task_id] = fisher

        # Store optimal parameters
        self.optimal_params[task_id] = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
        }

    def penalty(self) -> torch.Tensor:
        """
        Compute EWC penalty for current parameters.

        Returns:
            EWC regularization loss
        """
        if not self.fisher_matrices:
            return torch.tensor(0.0)

        penalty = 0.0

        for task_id in self.fisher_matrices:
            fisher = self.fisher_matrices[task_id]
            optimal = self.optimal_params[task_id]

            for name, param in self.model.named_parameters():
                if name in fisher:
                    penalty += (
                        fisher[name] *
                        (param - optimal[name]) ** 2
                    ).sum()

        return self.ewc_lambda * penalty / len(self.fisher_matrices)

    def forward(
        self,
        grid_a: torch.Tensor,
        scenario_a: torch.Tensor,
        grid_b: torch.Tensor,
        scenario_b: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through base model."""
        return self.model(grid_a, scenario_a, grid_b, scenario_b)


class ProgressiveNetwork(nn.Module):
    """
    Progressive Neural Network for continual learning.

    Adds new columns for new tasks while keeping old columns frozen.
    Uses lateral connections to transfer knowledge.
    """

    def __init__(
        self,
        config: RankingV2Config,
        n_initial_columns: int = 1,
    ):
        """
        Initialize progressive network.

        Args:
            config: Model configuration
            n_initial_columns: Number of initial columns
        """
        super().__init__()

        self.config = config

        # Columns (each is a full ranker)
        self.columns = nn.ModuleList([
            CrossAttentionRanker(config)
            for _ in range(n_initial_columns)
        ])

        # Lateral connections (from old columns to new)
        self.laterals = nn.ModuleList()

        # Track which column is active
        self.active_column = 0

    def add_column(self):
        """Add new column for new task."""
        new_column = CrossAttentionRanker(self.config)

        # Freeze all existing columns
        for col in self.columns:
            for param in col.parameters():
                param.requires_grad = False

        # Add lateral connections from all existing columns
        n_existing = len(self.columns)
        lateral = nn.ModuleList([
            nn.Linear(
                self.config.latent_dim,
                self.config.latent_dim,
            )
            for _ in range(n_existing)
        ])

        self.columns.append(new_column)
        self.laterals.append(lateral)
        self.active_column = len(self.columns) - 1

    def forward(
        self,
        grid_a: torch.Tensor,
        scenario_a: torch.Tensor,
        grid_b: torch.Tensor,
        scenario_b: torch.Tensor,
        column_idx: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through specified column.

        Args:
            grid_a, scenario_a, grid_b, scenario_b: Inputs
            column_idx: Which column to use (None = active)

        Returns:
            Model outputs
        """
        idx = column_idx if column_idx is not None else self.active_column

        # Get base outputs from target column
        outputs = self.columns[idx](grid_a, scenario_a, grid_b, scenario_b)

        # Add lateral connections for new columns
        if idx > 0 and idx < len(self.laterals) + 1:
            lateral_idx = idx - 1
            lateral_feats = torch.zeros_like(outputs['latent_a'])

            for i, lateral in enumerate(self.laterals[lateral_idx]):
                with torch.no_grad():
                    old_output = self.columns[i](
                        grid_a, scenario_a, grid_b, scenario_b
                    )
                lateral_feats += lateral(old_output['latent_a'])

            # Add to latent (could be more sophisticated)
            outputs['latent_a'] = outputs['latent_a'] + lateral_feats

        return outputs


class KnowledgeDistillation:
    """
    Knowledge distillation for continual learning.

    Uses a frozen teacher model to guide the student model,
    preventing forgetting of old knowledge.
    """

    def __init__(
        self,
        student: CrossAttentionRanker,
        temperature: float = 2.0,
        alpha: float = 0.5,
    ):
        """
        Initialize knowledge distillation.

        Args:
            student: Student model (being trained)
            temperature: Softmax temperature for distillation
            alpha: Weight for distillation loss
        """
        self.student = student
        self.teacher: Optional[CrossAttentionRanker] = None
        self.temperature = temperature
        self.alpha = alpha

    def update_teacher(self):
        """Update teacher with current student weights."""
        self.teacher = deepcopy(self.student)
        for param in self.teacher.parameters():
            param.requires_grad = False

    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute distillation loss.

        Args:
            student_logits: Student predictions
            teacher_logits: Teacher predictions

        Returns:
            Distillation loss
        """
        # Softened probabilities
        student_probs = torch.sigmoid(student_logits / self.temperature)
        teacher_probs = torch.sigmoid(teacher_logits / self.temperature)

        # KL divergence
        loss = F.binary_cross_entropy(
            student_probs,
            teacher_probs,
            reduction='mean',
        )

        return loss * (self.temperature ** 2)

    def compute_loss(
        self,
        grid_a: torch.Tensor,
        scenario_a: torch.Tensor,
        grid_b: torch.Tensor,
        scenario_b: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss (task loss + distillation).

        Args:
            Inputs and labels

        Returns:
            (total_loss, loss_dict)
        """
        # Student predictions
        student_outputs = self.student(grid_a, scenario_a, grid_b, scenario_b)
        student_logits = student_outputs['logit']

        # Task loss
        task_loss = F.binary_cross_entropy_with_logits(
            student_logits,
            labels.float(),
        )

        # Distillation loss (if teacher exists)
        if self.teacher is not None:
            with torch.no_grad():
                teacher_outputs = self.teacher(
                    grid_a, scenario_a, grid_b, scenario_b
                )
                teacher_logits = teacher_outputs['logit']

            distill_loss = self.distillation_loss(student_logits, teacher_logits)
            total_loss = (1 - self.alpha) * task_loss + self.alpha * distill_loss

            return total_loss, {
                'task_loss': task_loss.item(),
                'distill_loss': distill_loss.item(),
                'total_loss': total_loss.item(),
            }

        return task_loss, {
            'task_loss': task_loss.item(),
            'total_loss': task_loss.item(),
        }


class ContinualLearner:
    """
    Complete continual learning system.

    Combines multiple strategies for preventing catastrophic forgetting.
    """

    def __init__(
        self,
        model: CrossAttentionRanker,
        config: RankingV2Config,
        strategy: str = "replay_ewc",  # "replay", "ewc", "replay_ewc", "distill"
        replay_buffer_size: int = 10000,
        ewc_lambda: float = 100.0,
        replay_ratio: float = 0.5,
        device: torch.device = torch.device('cpu'),
    ):
        """
        Initialize continual learner.

        Args:
            model: Base ranking model
            config: Model configuration
            strategy: Continual learning strategy
            replay_buffer_size: Size of replay buffer
            ewc_lambda: EWC regularization strength
            replay_ratio: Fraction of batch from replay
            device: Device
        """
        self.model = model.to(device)
        self.config = config
        self.strategy = strategy
        self.replay_ratio = replay_ratio
        self.device = device

        # Components
        self.replay_buffer = ExperienceReplayBuffer(replay_buffer_size)

        if "ewc" in strategy:
            self.ewc = EWC(model, ewc_lambda)
        else:
            self.ewc = None

        if strategy == "distill":
            self.distillation = KnowledgeDistillation(model)
        else:
            self.distillation = None

        # State
        self.state = ContinualLearningState(
            current_task=0,
            tasks_seen=[],
            total_samples_seen=0,
            forgetting_measure={},
            forward_transfer={},
        )

    def train_on_task(
        self,
        task_id: int,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        epochs: int = 10,
        lr: float = 1e-4,
    ) -> Dict[str, List[float]]:
        """
        Train on a new task.

        Args:
            task_id: Task identifier
            train_loader: Training data
            val_loader: Validation data
            epochs: Number of epochs
            lr: Learning rate

        Returns:
            Training history
        """
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        history = {'train_loss': [], 'val_acc': [], 'ewc_penalty': []}

        # Update teacher before training (for distillation)
        if self.distillation is not None and self.state.tasks_seen:
            self.distillation.update_teacher()

        for epoch in range(epochs):
            total_loss = 0.0
            ewc_penalty = 0.0
            n_batches = 0

            for batch in train_loader:
                optimizer.zero_grad()

                # Move to device
                grid_a = batch['grid_a'].to(self.device)
                scenario_a = batch['scenario_a'].to(self.device)
                grid_b = batch['grid_b'].to(self.device)
                scenario_b = batch['scenario_b'].to(self.device)
                labels = batch['label'].to(self.device)

                # Forward pass
                if self.distillation is not None:
                    loss, loss_dict = self.distillation.compute_loss(
                        grid_a, scenario_a, grid_b, scenario_b, labels
                    )
                else:
                    outputs = self.model(
                        grid_a, scenario_a, grid_b, scenario_b
                    )
                    loss = F.binary_cross_entropy_with_logits(
                        outputs['logit'],
                        labels.float(),
                    )

                # Add replay samples
                if "replay" in self.strategy and len(self.replay_buffer) > 0:
                    replay_size = int(grid_a.size(0) * self.replay_ratio)
                    replay_exps = self.replay_buffer.sample(replay_size)
                    replay_batch = self.replay_buffer.collate(replay_exps)

                    replay_outputs = self.model(
                        replay_batch['grid_a'].to(self.device),
                        replay_batch['scenario_a'].to(self.device),
                        replay_batch['grid_b'].to(self.device),
                        replay_batch['scenario_b'].to(self.device),
                    )
                    replay_loss = F.binary_cross_entropy_with_logits(
                        replay_outputs['logit'],
                        replay_batch['label'].to(self.device).float(),
                    )
                    loss = loss + replay_loss

                # Add EWC penalty
                if self.ewc is not None and self.state.tasks_seen:
                    penalty = self.ewc.penalty()
                    loss = loss + penalty
                    ewc_penalty += penalty.item()

                # Backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                # Store in replay buffer
                self.replay_buffer.add_batch(
                    batch['grid_a'],
                    batch['scenario_a'],
                    batch['grid_b'],
                    batch['scenario_b'],
                    batch['label'],
                    task_id=task_id,
                )

                total_loss += loss.item()
                n_batches += 1

            history['train_loss'].append(total_loss / n_batches)
            history['ewc_penalty'].append(ewc_penalty / n_batches if n_batches > 0 else 0)

            # Validation
            if val_loader is not None:
                val_acc = self._evaluate(val_loader)
                history['val_acc'].append(val_acc)

        # Compute Fisher for EWC after training
        if self.ewc is not None:
            self.ewc.compute_fisher(train_loader, task_id, self.device)

        # Update state
        self.state.current_task = task_id
        if task_id not in self.state.tasks_seen:
            self.state.tasks_seen.append(task_id)

        return history

    def _evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
    ) -> float:
        """Evaluate accuracy on dataloader."""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                outputs = self.model(
                    batch['grid_a'].to(self.device),
                    batch['scenario_a'].to(self.device),
                    batch['grid_b'].to(self.device),
                    batch['scenario_b'].to(self.device),
                )
                pred = (outputs['logit'] > 0).long()
                correct += (pred == batch['label'].to(self.device)).sum().item()
                total += batch['label'].size(0)

        self.model.train()
        return correct / total if total > 0 else 0.0

    def evaluate_forgetting(
        self,
        task_loaders: Dict[int, torch.utils.data.DataLoader],
    ) -> Dict[int, float]:
        """
        Evaluate forgetting on all previous tasks.

        Args:
            task_loaders: Dict mapping task_id to validation loader

        Returns:
            Dict mapping task_id to accuracy
        """
        results = {}
        for task_id, loader in task_loaders.items():
            acc = self._evaluate(loader)
            results[task_id] = acc

            # Compare to previous performance
            if task_id in self.state.forgetting_measure:
                prev_acc = self.state.forgetting_measure[task_id]
                forgetting = prev_acc - acc
                self.state.forgetting_measure[task_id] = forgetting
            else:
                self.state.forgetting_measure[task_id] = acc

        return results

    def get_state(self) -> ContinualLearningState:
        """Get current learning state."""
        return self.state

    def save_checkpoint(self, path: str):
        """Save learner checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'state': self.state,
            'ewc_fisher': self.ewc.fisher_matrices if self.ewc else None,
            'ewc_optimal': self.ewc.optimal_params if self.ewc else None,
        }, path)

    def load_checkpoint(self, path: str):
        """Load learner checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.state = checkpoint['state']

        if self.ewc and checkpoint['ewc_fisher']:
            self.ewc.fisher_matrices = checkpoint['ewc_fisher']
            self.ewc.optimal_params = checkpoint['ewc_optimal']


class OnlineLearner:
    """
    Online learning for streaming data.

    Processes data sample-by-sample or in small batches.
    """

    def __init__(
        self,
        model: CrossAttentionRanker,
        lr: float = 1e-4,
        momentum: float = 0.9,
        buffer_size: int = 1000,
        device: torch.device = torch.device('cpu'),
    ):
        """
        Initialize online learner.

        Args:
            model: Ranking model
            lr: Learning rate
            momentum: SGD momentum
            buffer_size: Size of mini-buffer for batch updates
            device: Device
        """
        self.model = model.to(device)
        self.device = device

        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
        )

        self.buffer: List[Experience] = []
        self.buffer_size = buffer_size
        self.update_counter = 0

    def update(
        self,
        grid_a: torch.Tensor,
        scenario_a: torch.Tensor,
        grid_b: torch.Tensor,
        scenario_b: torch.Tensor,
        label: torch.Tensor,
    ) -> float:
        """
        Online update with single sample.

        Args:
            Single sample data

        Returns:
            Loss value
        """
        self.model.train()
        self.optimizer.zero_grad()

        outputs = self.model(
            grid_a.unsqueeze(0).to(self.device),
            scenario_a.unsqueeze(0).to(self.device),
            grid_b.unsqueeze(0).to(self.device),
            scenario_b.unsqueeze(0).to(self.device),
        )

        loss = F.binary_cross_entropy_with_logits(
            outputs['logit'],
            label.unsqueeze(0).to(self.device).float(),
        )

        loss.backward()
        self.optimizer.step()

        self.update_counter += 1
        return loss.item()

    def update_batch(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> float:
        """
        Online update with batch.

        Args:
            batch: Batch data dict

        Returns:
            Loss value
        """
        self.model.train()
        self.optimizer.zero_grad()

        outputs = self.model(
            batch['grid_a'].to(self.device),
            batch['scenario_a'].to(self.device),
            batch['grid_b'].to(self.device),
            batch['scenario_b'].to(self.device),
        )

        loss = F.binary_cross_entropy_with_logits(
            outputs['logit'],
            batch['label'].to(self.device).float(),
        )

        loss.backward()
        self.optimizer.step()

        self.update_counter += batch['label'].size(0)
        return loss.item()


def create_continual_learner(
    model: CrossAttentionRanker,
    config: RankingV2Config,
    strategy: str = "replay_ewc",
    device: torch.device = torch.device('cpu'),
    **kwargs,
) -> ContinualLearner:
    """
    Factory function to create continual learner.

    Args:
        model: Base model
        config: Configuration
        strategy: Learning strategy
        device: Device
        **kwargs: Additional arguments

    Returns:
        ContinualLearner instance
    """
    return ContinualLearner(
        model=model,
        config=config,
        strategy=strategy,
        replay_buffer_size=kwargs.get('replay_buffer_size', 10000),
        ewc_lambda=kwargs.get('ewc_lambda', 100.0),
        replay_ratio=kwargs.get('replay_ratio', 0.5),
        device=device,
    )

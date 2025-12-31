"""
Configuration Generator for Optimal Door/Exit Placement

Generates optimized door and exit configurations for floor plans
using the ranking model as a learned objective function.

Methods:
1. Evolutionary Algorithm: Genetic optimization of configurations
2. Monte Carlo Tree Search (MCTS): Tree-based configuration exploration
3. Gradient-based Optimization: Differentiable configuration generation
4. VAE-based Generation: Learned generative model for configurations

Key Idea: Use ranking model score as fitness function for optimization
"""

from typing import Dict, List, Optional, Tuple, Callable, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from copy import deepcopy
import heapq
import random

from .model import CrossAttentionRanker
from .config import RankingV2Config


@dataclass
class Configuration:
    """Represents a door/exit configuration for a floor plan."""
    floor_plan_id: str
    door_positions: List[Tuple[int, int]]
    exit_positions: List[Tuple[int, int]]
    score: Optional[float] = None
    metadata: Dict = field(default_factory=dict)

    def to_grid_encoding(self, grid_shape: Tuple[int, int]) -> np.ndarray:
        """
        Convert configuration to grid encoding.

        Args:
            grid_shape: (H, W) shape of the grid

        Returns:
            2-channel grid (doors, exits)
        """
        H, W = grid_shape
        encoding = np.zeros((2, H, W), dtype=np.float32)

        for y, x in self.door_positions:
            if 0 <= y < H and 0 <= x < W:
                encoding[0, y, x] = 1.0

        for y, x in self.exit_positions:
            if 0 <= y < H and 0 <= x < W:
                encoding[1, y, x] = 1.0

        return encoding

    def copy(self) -> 'Configuration':
        """Create a deep copy."""
        return Configuration(
            floor_plan_id=self.floor_plan_id,
            door_positions=list(self.door_positions),
            exit_positions=list(self.exit_positions),
            score=self.score,
            metadata=dict(self.metadata),
        )


@dataclass
class GenerationResult:
    """Result of configuration generation."""
    best_config: Configuration
    all_configs: List[Configuration]
    optimization_history: List[Dict]
    total_evaluations: int
    converged: bool


class ConfigurationScorer:
    """
    Scores configurations using the ranking model.

    Converts configurations to model inputs and evaluates them.
    """

    def __init__(
        self,
        model: CrossAttentionRanker,
        base_grid: torch.Tensor,
        scenario: torch.Tensor,
        device: torch.device = torch.device('cpu'),
    ):
        """
        Initialize scorer.

        Args:
            model: Ranking model
            base_grid: Base floor plan grid (3, H, W) - walls, passable, valid
            scenario: Scenario parameters (4,)
            device: Device for computation
        """
        self.model = model.to(device)
        self.model.eval()
        self.base_grid = base_grid.to(device)
        self.scenario = scenario.to(device)
        self.device = device

        self._call_count = 0

    def score(self, config: Configuration) -> float:
        """
        Score a single configuration.

        Args:
            config: Configuration to score

        Returns:
            Predicted score (higher = better)
        """
        grid = self._config_to_grid(config)

        with torch.no_grad():
            score = self.model.score_single(
                grid.unsqueeze(0),
                self.scenario.unsqueeze(0),
            )

        self._call_count += 1
        return score.item()

    def score_batch(self, configs: List[Configuration]) -> List[float]:
        """Score a batch of configurations."""
        grids = torch.stack([self._config_to_grid(c) for c in configs])
        scenarios = self.scenario.unsqueeze(0).expand(len(configs), -1)

        with torch.no_grad():
            scores = self.model.score_single(grids, scenarios)

        self._call_count += len(configs)
        return scores.tolist()

    def compare(
        self,
        config_a: Configuration,
        config_b: Configuration,
    ) -> float:
        """
        Compare two configurations.

        Returns:
            P(A > B) - probability that A is better than B
        """
        grid_a = self._config_to_grid(config_a)
        grid_b = self._config_to_grid(config_b)

        with torch.no_grad():
            outputs = self.model(
                grid_a.unsqueeze(0),
                self.scenario.unsqueeze(0),
                grid_b.unsqueeze(0),
                self.scenario.unsqueeze(0),
            )
            prob = torch.sigmoid(outputs['logit'])

        self._call_count += 1
        return prob.item()

    def _config_to_grid(self, config: Configuration) -> torch.Tensor:
        """Convert configuration to full grid tensor."""
        H, W = self.base_grid.shape[1], self.base_grid.shape[2]

        # Get door/exit encoding
        config_encoding = config.to_grid_encoding((H, W))
        config_tensor = torch.tensor(config_encoding, device=self.device)

        # Combine with base grid: [wall, passable, doors, exits, valid]
        grid = torch.zeros(5, H, W, device=self.device)
        grid[0] = self.base_grid[0]  # walls
        grid[1] = self.base_grid[1]  # passable
        grid[2] = config_tensor[0]   # doors
        grid[3] = config_tensor[1]   # exits
        grid[4] = self.base_grid[2]  # valid

        return grid

    @property
    def num_evaluations(self) -> int:
        """Number of model evaluations performed."""
        return self._call_count


class EvolutionaryOptimizer:
    """
    Evolutionary algorithm for configuration optimization.

    Uses genetic operations (crossover, mutation) to evolve
    a population of configurations toward optimality.
    """

    def __init__(
        self,
        scorer: ConfigurationScorer,
        valid_positions: List[Tuple[int, int]],
        n_doors: int = 3,
        n_exits: int = 2,
        population_size: int = 50,
        n_generations: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        elite_size: int = 5,
        tournament_size: int = 5,
    ):
        """
        Initialize evolutionary optimizer.

        Args:
            scorer: Configuration scorer
            valid_positions: List of valid positions for doors/exits
            n_doors: Number of doors to place
            n_exits: Number of exits to place
            population_size: Population size
            n_generations: Number of generations
            mutation_rate: Probability of mutation per position
            crossover_rate: Probability of crossover
            elite_size: Number of best individuals to preserve
            tournament_size: Tournament selection size
        """
        self.scorer = scorer
        self.valid_positions = valid_positions
        self.n_doors = n_doors
        self.n_exits = n_exits
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size

    def optimize(
        self,
        floor_plan_id: str,
        initial_population: Optional[List[Configuration]] = None,
    ) -> GenerationResult:
        """
        Run evolutionary optimization.

        Args:
            floor_plan_id: Floor plan identifier
            initial_population: Optional initial population

        Returns:
            GenerationResult with optimized configurations
        """
        # Initialize population
        if initial_population is not None:
            population = initial_population[:self.population_size]
            while len(population) < self.population_size:
                population.append(self._random_config(floor_plan_id))
        else:
            population = [
                self._random_config(floor_plan_id)
                for _ in range(self.population_size)
            ]

        # Score initial population
        for config in population:
            config.score = self.scorer.score(config)

        history = []
        best_ever = max(population, key=lambda c: c.score)

        for gen in range(self.n_generations):
            # Selection
            parents = self._tournament_selection(population)

            # Create offspring
            offspring = []

            for i in range(0, len(parents) - 1, 2):
                parent1, parent2 = parents[i], parents[i + 1]

                # Crossover
                if random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                # Mutation
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)

                offspring.extend([child1, child2])

            # Score offspring
            for config in offspring:
                config.score = self.scorer.score(config)

            # Elitism + selection
            combined = population + offspring
            combined.sort(key=lambda c: c.score, reverse=True)

            # Keep elite
            population = combined[:self.elite_size]

            # Fill rest with tournament selection from remaining
            remaining = combined[self.elite_size:]
            while len(population) < self.population_size and remaining:
                selected = self._tournament_select_one(remaining)
                population.append(selected)
                remaining.remove(selected)

            # Track best
            gen_best = max(population, key=lambda c: c.score)
            if gen_best.score > best_ever.score:
                best_ever = gen_best.copy()

            history.append({
                'generation': gen,
                'best_score': gen_best.score,
                'mean_score': np.mean([c.score for c in population]),
                'std_score': np.std([c.score for c in population]),
            })

            # Early stopping if converged
            if gen > 10:
                recent_best = [h['best_score'] for h in history[-10:]]
                if max(recent_best) - min(recent_best) < 0.001:
                    break

        return GenerationResult(
            best_config=best_ever,
            all_configs=sorted(population, key=lambda c: c.score, reverse=True),
            optimization_history=history,
            total_evaluations=self.scorer.num_evaluations,
            converged=len(history) < self.n_generations,
        )

    def _random_config(self, floor_plan_id: str) -> Configuration:
        """Generate random configuration."""
        positions = random.sample(self.valid_positions, self.n_doors + self.n_exits)
        return Configuration(
            floor_plan_id=floor_plan_id,
            door_positions=positions[:self.n_doors],
            exit_positions=positions[self.n_doors:],
        )

    def _tournament_selection(
        self,
        population: List[Configuration],
    ) -> List[Configuration]:
        """Select parents using tournament selection."""
        selected = []
        for _ in range(len(population)):
            selected.append(self._tournament_select_one(population))
        return selected

    def _tournament_select_one(
        self,
        population: List[Configuration],
    ) -> Configuration:
        """Select one individual via tournament."""
        tournament = random.sample(
            population,
            min(self.tournament_size, len(population)),
        )
        return max(tournament, key=lambda c: c.score).copy()

    def _crossover(
        self,
        parent1: Configuration,
        parent2: Configuration,
    ) -> Tuple[Configuration, Configuration]:
        """Single-point crossover."""
        # Crossover doors
        cut_door = random.randint(0, self.n_doors)
        doors1 = parent1.door_positions[:cut_door] + parent2.door_positions[cut_door:]
        doors2 = parent2.door_positions[:cut_door] + parent1.door_positions[cut_door:]

        # Crossover exits
        cut_exit = random.randint(0, self.n_exits)
        exits1 = parent1.exit_positions[:cut_exit] + parent2.exit_positions[cut_exit:]
        exits2 = parent2.exit_positions[:cut_exit] + parent1.exit_positions[cut_exit:]

        child1 = Configuration(
            floor_plan_id=parent1.floor_plan_id,
            door_positions=doors1[:self.n_doors],
            exit_positions=exits1[:self.n_exits],
        )
        child2 = Configuration(
            floor_plan_id=parent1.floor_plan_id,
            door_positions=doors2[:self.n_doors],
            exit_positions=exits2[:self.n_exits],
        )

        return child1, child2

    def _mutate(self, config: Configuration) -> Configuration:
        """Mutate configuration."""
        config = config.copy()

        # Mutate door positions
        for i in range(len(config.door_positions)):
            if random.random() < self.mutation_rate:
                new_pos = random.choice(self.valid_positions)
                config.door_positions[i] = new_pos

        # Mutate exit positions
        for i in range(len(config.exit_positions)):
            if random.random() < self.mutation_rate:
                new_pos = random.choice(self.valid_positions)
                config.exit_positions[i] = new_pos

        return config


class MCTSNode:
    """Node in the Monte Carlo Tree Search."""

    def __init__(
        self,
        config: Configuration,
        parent: Optional['MCTSNode'] = None,
    ):
        self.config = config
        self.parent = parent
        self.children: List['MCTSNode'] = []
        self.visits = 0
        self.total_score = 0.0
        self.is_terminal = False

    @property
    def mean_score(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.total_score / self.visits

    def ucb1(self, exploration_weight: float = 1.414) -> float:
        """Upper Confidence Bound for Trees."""
        if self.visits == 0:
            return float('inf')

        parent_visits = self.parent.visits if self.parent else 1
        exploitation = self.mean_score
        exploration = exploration_weight * np.sqrt(np.log(parent_visits) / self.visits)

        return exploitation + exploration


class MCTSOptimizer:
    """
    Monte Carlo Tree Search for configuration optimization.

    Treats configuration space as a tree and uses UCB1 for
    exploration/exploitation tradeoff.
    """

    def __init__(
        self,
        scorer: ConfigurationScorer,
        valid_positions: List[Tuple[int, int]],
        n_doors: int = 3,
        n_exits: int = 2,
        n_iterations: int = 1000,
        exploration_weight: float = 1.414,
    ):
        """
        Initialize MCTS optimizer.

        Args:
            scorer: Configuration scorer
            valid_positions: Valid positions for doors/exits
            n_doors: Number of doors
            n_exits: Number of exits
            n_iterations: Number of MCTS iterations
            exploration_weight: UCB1 exploration parameter
        """
        self.scorer = scorer
        self.valid_positions = valid_positions
        self.n_doors = n_doors
        self.n_exits = n_exits
        self.n_iterations = n_iterations
        self.exploration_weight = exploration_weight

    def optimize(self, floor_plan_id: str) -> GenerationResult:
        """Run MCTS optimization."""
        # Create root with empty configuration
        root_config = Configuration(
            floor_plan_id=floor_plan_id,
            door_positions=[],
            exit_positions=[],
        )
        root = MCTSNode(root_config)

        history = []
        best_config = None
        best_score = float('-inf')

        for iteration in range(self.n_iterations):
            # Selection
            node = self._select(root)

            # Expansion
            if not node.is_terminal and node.visits > 0:
                node = self._expand(node)

            # Simulation (rollout)
            score = self._simulate(node)

            # Backpropagation
            self._backpropagate(node, score)

            # Track best
            if score > best_score:
                best_score = score
                best_config = node.config.copy()
                best_config.score = score

            if (iteration + 1) % 100 == 0:
                history.append({
                    'iteration': iteration + 1,
                    'best_score': best_score,
                    'root_visits': root.visits,
                    'num_children': len(root.children),
                })

        # Collect top configurations
        all_configs = self._collect_configs(root)
        all_configs.sort(key=lambda c: c.score if c.score else 0, reverse=True)

        return GenerationResult(
            best_config=best_config,
            all_configs=all_configs[:50],
            optimization_history=history,
            total_evaluations=self.scorer.num_evaluations,
            converged=True,
        )

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select node using UCB1."""
        while node.children:
            node = max(node.children, key=lambda n: n.ucb1(self.exploration_weight))
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expand node by adding a new position."""
        config = node.config

        # Determine what to add next
        total_placed = len(config.door_positions) + len(config.exit_positions)

        if total_placed >= self.n_doors + self.n_exits:
            node.is_terminal = True
            return node

        # Add a new position
        used_positions = set(config.door_positions + config.exit_positions)
        available = [p for p in self.valid_positions if p not in used_positions]

        if not available:
            node.is_terminal = True
            return node

        # Choose random position for expansion
        new_pos = random.choice(available)

        new_config = config.copy()
        if len(new_config.door_positions) < self.n_doors:
            new_config.door_positions.append(new_pos)
        else:
            new_config.exit_positions.append(new_pos)

        child = MCTSNode(new_config, parent=node)
        node.children.append(child)

        return child

    def _simulate(self, node: MCTSNode) -> float:
        """Random rollout from node to terminal state."""
        config = node.config.copy()

        # Complete configuration randomly
        used = set(config.door_positions + config.exit_positions)
        available = [p for p in self.valid_positions if p not in used]

        while len(config.door_positions) < self.n_doors and available:
            pos = random.choice(available)
            config.door_positions.append(pos)
            available.remove(pos)

        while len(config.exit_positions) < self.n_exits and available:
            pos = random.choice(available)
            config.exit_positions.append(pos)
            available.remove(pos)

        # Score the complete configuration
        return self.scorer.score(config)

    def _backpropagate(self, node: MCTSNode, score: float):
        """Backpropagate score up the tree."""
        while node is not None:
            node.visits += 1
            node.total_score += score
            node = node.parent

    def _collect_configs(self, node: MCTSNode) -> List[Configuration]:
        """Collect all terminal configurations from tree."""
        configs = []

        def dfs(n):
            if n.is_terminal and n.config.score is not None:
                configs.append(n.config)
            for child in n.children:
                dfs(child)

        dfs(node)
        return configs


class ConfigurationVAE(nn.Module):
    """
    Variational Autoencoder for configuration generation.

    Learns a latent space of configurations and can generate
    new configurations by sampling.
    """

    def __init__(
        self,
        grid_shape: Tuple[int, int],
        latent_dim: int = 32,
        hidden_dim: int = 256,
    ):
        """
        Initialize VAE.

        Args:
            grid_shape: (H, W) shape of configuration grid
            latent_dim: Dimension of latent space
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.grid_shape = grid_shape
        self.latent_dim = latent_dim
        H, W = grid_shape
        input_dim = 2 * H * W  # 2 channels: doors, exits

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),  # Output probabilities
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode to latent distribution."""
        h = self.encoder(x.view(x.size(0), -1))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent to configuration grid."""
        out = self.decoder(z)
        H, W = self.grid_shape
        return out.view(-1, 2, H, W)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def sample(self, n_samples: int = 1) -> torch.Tensor:
        """Sample new configurations from prior."""
        z = torch.randn(n_samples, self.latent_dim)
        return self.decode(z)

    def loss_function(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute VAE loss.

        Args:
            recon_x: Reconstructed configuration
            x: Original configuration
            mu: Latent mean
            logvar: Latent log variance
            beta: KL divergence weight

        Returns:
            (loss, loss_dict)
        """
        # Reconstruction loss (BCE)
        recon_loss = F.binary_cross_entropy(
            recon_x.view(-1), x.view(-1), reduction='sum'
        )

        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        total_loss = recon_loss + beta * kl_loss

        return total_loss, {
            'reconstruction': recon_loss.item(),
            'kl': kl_loss.item(),
            'total': total_loss.item(),
        }


class GradientOptimizer:
    """
    Gradient-based optimization for differentiable configuration generation.

    Uses soft relaxation of discrete positions and optimizes via gradients.
    """

    def __init__(
        self,
        scorer: ConfigurationScorer,
        grid_shape: Tuple[int, int],
        n_doors: int = 3,
        n_exits: int = 2,
        n_iterations: int = 100,
        learning_rate: float = 0.1,
        temperature: float = 0.1,
    ):
        """
        Initialize gradient optimizer.

        Args:
            scorer: Configuration scorer (model)
            grid_shape: (H, W) shape of grid
            n_doors: Number of doors
            n_exits: Number of exits
            n_iterations: Optimization iterations
            learning_rate: Learning rate
            temperature: Softmax temperature for discretization
        """
        self.scorer = scorer
        self.grid_shape = grid_shape
        self.n_doors = n_doors
        self.n_exits = n_exits
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.temperature = temperature

    def optimize(self, floor_plan_id: str) -> GenerationResult:
        """
        Run gradient-based optimization.

        Uses Gumbel-Softmax for differentiable sampling.
        """
        H, W = self.grid_shape
        device = self.scorer.device

        # Initialize logits for door and exit positions
        door_logits = nn.Parameter(torch.randn(self.n_doors, H * W, device=device))
        exit_logits = nn.Parameter(torch.randn(self.n_exits, H * W, device=device))

        optimizer = torch.optim.Adam([door_logits, exit_logits], lr=self.learning_rate)

        history = []
        best_score = float('-inf')
        best_config = None

        for iteration in range(self.n_iterations):
            optimizer.zero_grad()

            # Gumbel-Softmax sampling
            door_probs = F.gumbel_softmax(door_logits, tau=self.temperature, hard=False)
            exit_probs = F.gumbel_softmax(exit_logits, tau=self.temperature, hard=False)

            # Reshape to grid
            door_grid = door_probs.view(self.n_doors, H, W).sum(dim=0)
            exit_grid = exit_probs.view(self.n_exits, H, W).sum(dim=0)

            # Clamp to [0, 1]
            door_grid = door_grid.clamp(0, 1)
            exit_grid = exit_grid.clamp(0, 1)

            # Construct full grid
            grid = torch.zeros(5, H, W, device=device)
            grid[0] = self.scorer.base_grid[0]  # walls
            grid[1] = self.scorer.base_grid[1]  # passable
            grid[2] = door_grid
            grid[3] = exit_grid
            grid[4] = self.scorer.base_grid[2]  # valid

            # Score (need to enable gradients in model temporarily)
            self.scorer.model.train()
            score = self.scorer.model.score_single(
                grid.unsqueeze(0),
                self.scorer.scenario.unsqueeze(0),
            )
            self.scorer.model.eval()

            # Maximize score (minimize negative score)
            loss = -score

            loss.backward()
            optimizer.step()

            # Extract discrete configuration
            with torch.no_grad():
                door_positions = self._extract_positions(door_logits, self.n_doors, H, W)
                exit_positions = self._extract_positions(exit_logits, self.n_exits, H, W)

                config = Configuration(
                    floor_plan_id=floor_plan_id,
                    door_positions=door_positions,
                    exit_positions=exit_positions,
                )
                config.score = self.scorer.score(config)

                if config.score > best_score:
                    best_score = config.score
                    best_config = config.copy()

            history.append({
                'iteration': iteration,
                'soft_score': score.item(),
                'discrete_score': config.score,
                'temperature': self.temperature,
            })

            # Anneal temperature
            self.temperature = max(0.01, self.temperature * 0.99)

        return GenerationResult(
            best_config=best_config,
            all_configs=[best_config],
            optimization_history=history,
            total_evaluations=self.scorer.num_evaluations,
            converged=True,
        )

    def _extract_positions(
        self,
        logits: torch.Tensor,
        n_items: int,
        H: int,
        W: int,
    ) -> List[Tuple[int, int]]:
        """Extract discrete positions from logits."""
        positions = []
        for i in range(n_items):
            idx = logits[i].argmax().item()
            y, x = idx // W, idx % W
            positions.append((y, x))
        return positions


def create_optimizer(
    method: str,
    scorer: ConfigurationScorer,
    valid_positions: List[Tuple[int, int]],
    **kwargs,
) -> Union[EvolutionaryOptimizer, MCTSOptimizer, GradientOptimizer]:
    """
    Factory function to create configuration optimizer.

    Args:
        method: Optimization method ('evolutionary', 'mcts', 'gradient')
        scorer: Configuration scorer
        valid_positions: Valid positions for doors/exits
        **kwargs: Method-specific arguments

    Returns:
        Optimizer instance
    """
    if method == 'evolutionary':
        return EvolutionaryOptimizer(
            scorer=scorer,
            valid_positions=valid_positions,
            n_doors=kwargs.get('n_doors', 3),
            n_exits=kwargs.get('n_exits', 2),
            population_size=kwargs.get('population_size', 50),
            n_generations=kwargs.get('n_generations', 100),
        )
    elif method == 'mcts':
        return MCTSOptimizer(
            scorer=scorer,
            valid_positions=valid_positions,
            n_doors=kwargs.get('n_doors', 3),
            n_exits=kwargs.get('n_exits', 2),
            n_iterations=kwargs.get('n_iterations', 1000),
        )
    elif method == 'gradient':
        return GradientOptimizer(
            scorer=scorer,
            grid_shape=kwargs.get('grid_shape', (96, 128)),
            n_doors=kwargs.get('n_doors', 3),
            n_exits=kwargs.get('n_exits', 2),
            n_iterations=kwargs.get('n_iterations', 100),
        )
    else:
        raise ValueError(f"Unknown optimization method: {method}")

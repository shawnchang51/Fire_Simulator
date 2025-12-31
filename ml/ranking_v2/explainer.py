"""
Explainable Recommendation Report Generator

Generates human-readable explanations for model predictions and recommendations.
Translates model internals (attention weights, feature importance, gradients)
into actionable insights for building safety professionals.

Components:
1. Feature Attribution: What input features drive the prediction
2. Attention Analysis: Which spatial regions are important
3. Counterfactual Explanations: How to improve a configuration
4. Natural Language Generation: Convert insights to readable text
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import json

from .model import CrossAttentionRanker
from .config import RankingV2Config
from .visualize import GradCAM


class ExplanationType(Enum):
    """Types of explanations that can be generated."""
    RANKING_COMPARISON = "ranking_comparison"
    SINGLE_CONFIG = "single_config"
    IMPROVEMENT_SUGGESTION = "improvement_suggestion"
    UNCERTAINTY_ANALYSIS = "uncertainty_analysis"


@dataclass
class SpatialAttention:
    """Spatial attention information for explanation."""
    heatmap: np.ndarray  # (H, W) attention weights
    hotspots: List[Dict]  # List of {position, importance, description}
    bottlenecks: List[Dict]  # Identified evacuation bottlenecks


@dataclass
class FeatureImportance:
    """Feature importance breakdown."""
    door_importance: Dict[Tuple[int, int], float]  # Position -> importance
    exit_importance: Dict[Tuple[int, int], float]
    scenario_importance: Dict[str, float]
    structural_importance: Dict[str, float]


@dataclass
class Recommendation:
    """Single recommendation for improvement."""
    action: str  # "add", "move", "remove"
    element_type: str  # "door", "exit"
    current_position: Optional[Tuple[int, int]]
    suggested_position: Optional[Tuple[int, int]]
    expected_improvement: float
    confidence: float
    reasoning: str


@dataclass
class ExplanationReport:
    """Complete explanation report for a prediction."""
    explanation_type: ExplanationType
    summary: str
    score: float
    confidence: float
    spatial_attention: Optional[SpatialAttention]
    feature_importance: Optional[FeatureImportance]
    recommendations: List[Recommendation]
    detailed_analysis: Dict[str, any]
    visualization_paths: Dict[str, str] = field(default_factory=dict)


class FeatureAttributor:
    """
    Computes feature attributions using integrated gradients.

    Attributes prediction to input features to understand
    what drives the model's decision.
    """

    def __init__(
        self,
        model: CrossAttentionRanker,
        device: torch.device = torch.device('cpu'),
        n_steps: int = 50,
    ):
        """
        Initialize attributor.

        Args:
            model: Ranking model
            device: Device for computation
            n_steps: Number of steps for integrated gradients
        """
        self.model = model.to(device)
        self.device = device
        self.n_steps = n_steps

    def compute_attributions(
        self,
        grid: torch.Tensor,
        scenario: torch.Tensor,
        target: str = "score",
    ) -> Dict[str, torch.Tensor]:
        """
        Compute feature attributions using integrated gradients.

        Args:
            grid: Input grid (5, H, W)
            scenario: Scenario parameters (4,)
            target: What to attribute to ("score", "survival_rate", etc.)

        Returns:
            Dict of attributions for each input
        """
        grid = grid.unsqueeze(0).to(self.device)
        scenario = scenario.unsqueeze(0).to(self.device)

        grid.requires_grad = True
        scenario.requires_grad = True

        # Baseline (zero input)
        baseline_grid = torch.zeros_like(grid)
        baseline_scenario = torch.zeros_like(scenario)

        # Integrated gradients
        grid_attributions = torch.zeros_like(grid)
        scenario_attributions = torch.zeros_like(scenario)

        for step in range(self.n_steps):
            alpha = step / self.n_steps

            # Interpolate between baseline and input
            interp_grid = baseline_grid + alpha * (grid - baseline_grid)
            interp_scenario = baseline_scenario + alpha * (scenario - baseline_scenario)

            interp_grid.requires_grad = True
            interp_scenario.requires_grad = True

            # Forward pass
            if target == "score":
                output = self.model.score_single(interp_grid, interp_scenario)
            else:
                # Auxiliary prediction
                aux_outputs = self.model.predict_auxiliary(interp_grid)
                output = aux_outputs.get(target, aux_outputs.get('survival_rate'))

            # Backward pass
            self.model.zero_grad()
            output.sum().backward()

            # Accumulate gradients
            if interp_grid.grad is not None:
                grid_attributions += interp_grid.grad / self.n_steps
            if interp_scenario.grad is not None:
                scenario_attributions += interp_scenario.grad / self.n_steps

        # Scale by input
        grid_attributions = grid_attributions * (grid - baseline_grid)
        scenario_attributions = scenario_attributions * (scenario - baseline_scenario)

        return {
            'grid': grid_attributions.squeeze(0).detach().cpu(),
            'scenario': scenario_attributions.squeeze(0).detach().cpu(),
        }

    def get_feature_importance(
        self,
        grid: torch.Tensor,
        scenario: torch.Tensor,
    ) -> FeatureImportance:
        """
        Get structured feature importance.

        Args:
            grid: Input grid (5, H, W)
            scenario: Scenario parameters (4,)

        Returns:
            FeatureImportance object
        """
        attributions = self.compute_attributions(grid, scenario)

        grid_attr = attributions['grid'].numpy()
        scenario_attr = attributions['scenario'].numpy()

        # Door importance (channel 2)
        door_channel = grid_attr[2]
        door_positions = np.argwhere(grid.numpy()[2] > 0.5)
        door_importance = {}
        for pos in door_positions:
            y, x = pos
            importance = float(door_channel[y, x])
            door_importance[(int(y), int(x))] = importance

        # Exit importance (channel 3)
        exit_channel = grid_attr[3]
        exit_positions = np.argwhere(grid.numpy()[3] > 0.5)
        exit_importance = {}
        for pos in exit_positions:
            y, x = pos
            importance = float(exit_channel[y, x])
            exit_importance[(int(y), int(x))] = importance

        # Scenario importance
        scenario_names = ['agent_count', 'num_fires', 'fire_spread_rate', 'fire_discovery_delay']
        scenario_importance = {
            name: float(scenario_attr[i])
            for i, name in enumerate(scenario_names)
        }

        # Structural importance (walls, passable)
        structural_importance = {
            'wall_layout': float(grid_attr[0].sum()),
            'passable_area': float(grid_attr[1].sum()),
        }

        return FeatureImportance(
            door_importance=door_importance,
            exit_importance=exit_importance,
            scenario_importance=scenario_importance,
            structural_importance=structural_importance,
        )


class AttentionAnalyzer:
    """
    Analyzes attention patterns to identify important spatial regions.
    """

    def __init__(
        self,
        model: CrossAttentionRanker,
        device: torch.device = torch.device('cpu'),
    ):
        """
        Initialize analyzer.

        Args:
            model: Ranking model with cross-attention
            device: Device for computation
        """
        self.model = model.to(device)
        self.device = device
        self.gradcam = GradCAM(model)

    def analyze_attention(
        self,
        grid: torch.Tensor,
        scenario: torch.Tensor,
    ) -> SpatialAttention:
        """
        Analyze spatial attention patterns.

        Args:
            grid: Input grid (5, H, W)
            scenario: Scenario parameters (4,)

        Returns:
            SpatialAttention object
        """
        grid = grid.unsqueeze(0).to(self.device)
        scenario = scenario.unsqueeze(0).to(self.device)

        # Get Grad-CAM heatmap
        heatmap = self.gradcam(grid, scenario)
        heatmap = heatmap.squeeze().cpu().numpy()

        # Identify hotspots (high attention regions)
        hotspots = self._find_hotspots(heatmap, grid.squeeze().cpu().numpy())

        # Identify bottlenecks
        bottlenecks = self._find_bottlenecks(heatmap, grid.squeeze().cpu().numpy())

        return SpatialAttention(
            heatmap=heatmap,
            hotspots=hotspots,
            bottlenecks=bottlenecks,
        )

    def _find_hotspots(
        self,
        heatmap: np.ndarray,
        grid: np.ndarray,
    ) -> List[Dict]:
        """Find attention hotspots in the heatmap."""
        from scipy import ndimage

        # Threshold for hotspots
        threshold = np.percentile(heatmap, 90)
        hotspot_mask = heatmap > threshold

        # Label connected components
        labeled, n_regions = ndimage.label(hotspot_mask)

        hotspots = []
        for region_id in range(1, n_regions + 1):
            region_mask = (labeled == region_id)
            y_coords, x_coords = np.where(region_mask)

            if len(y_coords) == 0:
                continue

            centroid_y = int(y_coords.mean())
            centroid_x = int(x_coords.mean())
            importance = float(heatmap[region_mask].mean())

            # Determine what's at this location
            description = self._describe_location(centroid_y, centroid_x, grid)

            hotspots.append({
                'position': (centroid_y, centroid_x),
                'importance': importance,
                'size': int(region_mask.sum()),
                'description': description,
            })

        # Sort by importance
        hotspots.sort(key=lambda x: x['importance'], reverse=True)
        return hotspots[:5]  # Top 5

    def _find_bottlenecks(
        self,
        heatmap: np.ndarray,
        grid: np.ndarray,
    ) -> List[Dict]:
        """Find potential evacuation bottlenecks."""
        from scipy import ndimage

        # Bottlenecks are narrow passable regions with high attention
        passable = grid[1] > 0.5
        walls = grid[0] > 0.5

        # Find narrow corridors using morphological operations
        dilated = ndimage.binary_dilation(passable, iterations=2)
        eroded = ndimage.binary_erosion(passable, iterations=2)
        narrow = passable & ~eroded

        # High attention narrow regions are bottlenecks
        bottleneck_mask = narrow & (heatmap > np.percentile(heatmap, 75))

        labeled, n_regions = ndimage.label(bottleneck_mask)

        bottlenecks = []
        for region_id in range(1, n_regions + 1):
            region_mask = (labeled == region_id)
            y_coords, x_coords = np.where(region_mask)

            if len(y_coords) < 3:  # Too small
                continue

            centroid_y = int(y_coords.mean())
            centroid_x = int(x_coords.mean())
            severity = float(heatmap[region_mask].mean())

            bottlenecks.append({
                'position': (centroid_y, centroid_x),
                'severity': severity,
                'width': self._estimate_corridor_width(centroid_y, centroid_x, passable),
                'description': f"Narrow corridor at ({centroid_x}, {centroid_y})",
            })

        bottlenecks.sort(key=lambda x: x['severity'], reverse=True)
        return bottlenecks[:3]  # Top 3

    def _describe_location(
        self,
        y: int,
        x: int,
        grid: np.ndarray,
    ) -> str:
        """Generate description of a location."""
        has_door = grid[2, y, x] > 0.5
        has_exit = grid[3, y, x] > 0.5
        is_passable = grid[1, y, x] > 0.5
        near_wall = self._near_wall(y, x, grid[0])

        if has_exit:
            return f"Exit at ({x}, {y})"
        elif has_door:
            return f"Door at ({x}, {y})"
        elif near_wall:
            return f"Wall-adjacent area at ({x}, {y})"
        elif is_passable:
            return f"Open corridor/room at ({x}, {y})"
        else:
            return f"Location ({x}, {y})"

    def _near_wall(self, y: int, x: int, walls: np.ndarray) -> bool:
        """Check if position is adjacent to a wall."""
        H, W = walls.shape
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W:
                    if walls[ny, nx] > 0.5:
                        return True
        return False

    def _estimate_corridor_width(
        self,
        y: int,
        x: int,
        passable: np.ndarray,
    ) -> int:
        """Estimate width of corridor at position."""
        H, W = passable.shape

        # Check horizontal and vertical widths
        h_width = 0
        for dx in range(-10, 11):
            if 0 <= x + dx < W and passable[y, x + dx]:
                h_width += 1

        v_width = 0
        for dy in range(-10, 11):
            if 0 <= y + dy < H and passable[y + dy, x]:
                v_width += 1

        return min(h_width, v_width)


class CounterfactualExplainer:
    """
    Generates counterfactual explanations.

    Answers "How could this configuration be improved?"
    """

    def __init__(
        self,
        model: CrossAttentionRanker,
        device: torch.device = torch.device('cpu'),
    ):
        """
        Initialize explainer.

        Args:
            model: Ranking model
            device: Device for computation
        """
        self.model = model.to(device)
        self.device = device

    def generate_recommendations(
        self,
        grid: torch.Tensor,
        scenario: torch.Tensor,
        valid_positions: List[Tuple[int, int]],
        n_recommendations: int = 5,
    ) -> List[Recommendation]:
        """
        Generate improvement recommendations.

        Args:
            grid: Current configuration grid (5, H, W)
            scenario: Scenario parameters (4,)
            valid_positions: Valid positions for doors/exits
            n_recommendations: Number of recommendations to generate

        Returns:
            List of Recommendation objects
        """
        grid = grid.to(self.device)
        scenario = scenario.to(self.device)

        # Get current score
        with torch.no_grad():
            current_score = self.model.score_single(
                grid.unsqueeze(0),
                scenario.unsqueeze(0),
            ).item()

        recommendations = []

        # Get current door and exit positions
        door_positions = list(zip(*np.where(grid[2].cpu().numpy() > 0.5)))
        exit_positions = list(zip(*np.where(grid[3].cpu().numpy() > 0.5)))

        # Try moving each door
        for i, (dy, dx) in enumerate(door_positions):
            best_new_pos = None
            best_improvement = 0

            for new_y, new_x in valid_positions:
                if (new_y, new_x) in door_positions or (new_y, new_x) in exit_positions:
                    continue

                # Create modified grid
                new_grid = grid.clone()
                new_grid[2, dy, dx] = 0
                new_grid[2, new_y, new_x] = 1

                with torch.no_grad():
                    new_score = self.model.score_single(
                        new_grid.unsqueeze(0),
                        scenario.unsqueeze(0),
                    ).item()

                improvement = new_score - current_score
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_new_pos = (new_y, new_x)

            if best_new_pos and best_improvement > 0:
                recommendations.append(Recommendation(
                    action="move",
                    element_type="door",
                    current_position=(int(dy), int(dx)),
                    suggested_position=best_new_pos,
                    expected_improvement=best_improvement,
                    confidence=min(1.0, best_improvement * 5),  # Heuristic confidence
                    reasoning=f"Moving door from ({dx}, {dy}) to ({best_new_pos[1]}, {best_new_pos[0]}) "
                             f"improves evacuation flow by {best_improvement:.3f}",
                ))

        # Try moving each exit
        for i, (ey, ex) in enumerate(exit_positions):
            best_new_pos = None
            best_improvement = 0

            for new_y, new_x in valid_positions[:50]:  # Limit search
                if (new_y, new_x) in door_positions or (new_y, new_x) in exit_positions:
                    continue

                new_grid = grid.clone()
                new_grid[3, ey, ex] = 0
                new_grid[3, new_y, new_x] = 1

                with torch.no_grad():
                    new_score = self.model.score_single(
                        new_grid.unsqueeze(0),
                        scenario.unsqueeze(0),
                    ).item()

                improvement = new_score - current_score
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_new_pos = (new_y, new_x)

            if best_new_pos and best_improvement > 0:
                recommendations.append(Recommendation(
                    action="move",
                    element_type="exit",
                    current_position=(int(ey), int(ex)),
                    suggested_position=best_new_pos,
                    expected_improvement=best_improvement,
                    confidence=min(1.0, best_improvement * 5),
                    reasoning=f"Relocating exit from ({ex}, {ey}) to ({best_new_pos[1]}, {best_new_pos[0]}) "
                             f"reduces evacuation time by {best_improvement:.3f}",
                ))

        # Sort by expected improvement
        recommendations.sort(key=lambda r: r.expected_improvement, reverse=True)
        return recommendations[:n_recommendations]


class NaturalLanguageGenerator:
    """
    Generates natural language explanations from structured data.
    """

    def __init__(self, language: str = "en"):
        """
        Initialize generator.

        Args:
            language: Language code ("en", "zh")
        """
        self.language = language
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, Dict[str, str]]:
        """Load language templates."""
        return {
            "en": {
                "ranking_better": "Configuration A is predicted to perform better than Configuration B.",
                "ranking_worse": "Configuration B is predicted to outperform Configuration A.",
                "high_confidence": "The model is highly confident ({confidence:.1%}) in this prediction.",
                "low_confidence": "There is some uncertainty in this prediction (confidence: {confidence:.1%}).",
                "bottleneck": "A potential bottleneck was identified at position ({x}, {y}) - {description}.",
                "door_important": "The door at ({x}, {y}) is critical for evacuation flow.",
                "exit_important": "The exit at ({x}, {y}) significantly impacts survival rates.",
                "recommendation_move": "Consider {action}ing the {element} from ({from_x}, {from_y}) to ({to_x}, {to_y}). Expected improvement: {improvement:.1%}.",
                "summary": "Based on analysis of the floor plan configuration and {n_agents} agents with {n_fires} fire sources, {main_finding}",
            },
            "zh": {
                "ranking_better": "預測配置 A 的表現優於配置 B。",
                "ranking_worse": "預測配置 B 的表現優於配置 A。",
                "high_confidence": "模型對此預測有高度信心（{confidence:.1%}）。",
                "low_confidence": "此預測存在一定不確定性（信心度：{confidence:.1%}）。",
                "bottleneck": "在位置 ({x}, {y}) 識別到潛在瓶頸 - {description}。",
                "door_important": "位於 ({x}, {y}) 的門對疏散流程至關重要。",
                "exit_important": "位於 ({x}, {y}) 的出口顯著影響存活率。",
                "recommendation_move": "建議將{element}從 ({from_x}, {from_y}) {action}至 ({to_x}, {to_y})。預期改善：{improvement:.1%}。",
                "summary": "根據樓層平面配置分析，在 {n_agents} 名人員和 {n_fires} 個火源的情況下，{main_finding}",
            },
        }

    def generate_summary(
        self,
        report: 'ExplanationReport',
        scenario: Optional[Dict] = None,
    ) -> str:
        """
        Generate natural language summary of explanation.

        Args:
            report: ExplanationReport to summarize
            scenario: Optional scenario parameters

        Returns:
            Natural language summary string
        """
        templates = self.templates[self.language]
        parts = []

        # Main finding
        if report.explanation_type == ExplanationType.RANKING_COMPARISON:
            if report.score > 0:
                parts.append(templates["ranking_better"])
            else:
                parts.append(templates["ranking_worse"])

        # Confidence
        if report.confidence > 0.8:
            parts.append(templates["high_confidence"].format(confidence=report.confidence))
        elif report.confidence < 0.6:
            parts.append(templates["low_confidence"].format(confidence=report.confidence))

        # Bottlenecks
        if report.spatial_attention and report.spatial_attention.bottlenecks:
            for bn in report.spatial_attention.bottlenecks[:2]:
                parts.append(templates["bottleneck"].format(
                    x=bn['position'][1],
                    y=bn['position'][0],
                    description=bn['description'],
                ))

        # Feature importance highlights
        if report.feature_importance:
            # Most important door
            if report.feature_importance.door_importance:
                top_door = max(
                    report.feature_importance.door_importance.items(),
                    key=lambda x: abs(x[1])
                )
                if abs(top_door[1]) > 0.1:
                    parts.append(templates["door_important"].format(
                        x=top_door[0][1],
                        y=top_door[0][0],
                    ))

            # Most important exit
            if report.feature_importance.exit_importance:
                top_exit = max(
                    report.feature_importance.exit_importance.items(),
                    key=lambda x: abs(x[1])
                )
                if abs(top_exit[1]) > 0.1:
                    parts.append(templates["exit_important"].format(
                        x=top_exit[0][1],
                        y=top_exit[0][0],
                    ))

        # Top recommendations
        for rec in report.recommendations[:2]:
            element = "door" if rec.element_type == "door" else "exit"
            if self.language == "zh":
                element = "門" if rec.element_type == "door" else "出口"

            parts.append(templates["recommendation_move"].format(
                action=rec.action,
                element=element,
                from_x=rec.current_position[1] if rec.current_position else 0,
                from_y=rec.current_position[0] if rec.current_position else 0,
                to_x=rec.suggested_position[1] if rec.suggested_position else 0,
                to_y=rec.suggested_position[0] if rec.suggested_position else 0,
                improvement=rec.expected_improvement,
            ))

        return " ".join(parts)


class ExplanationPipeline:
    """
    Complete pipeline for generating explanation reports.

    Combines all explanation components into a unified interface.
    """

    def __init__(
        self,
        model: CrossAttentionRanker,
        config: RankingV2Config,
        device: torch.device = torch.device('cpu'),
        language: str = "en",
    ):
        """
        Initialize pipeline.

        Args:
            model: Ranking model
            config: Model configuration
            device: Device for computation
            language: Language for natural language generation
        """
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Components
        self.attributor = FeatureAttributor(model, device)
        self.attention_analyzer = AttentionAnalyzer(model, device)
        self.counterfactual = CounterfactualExplainer(model, device)
        self.nlg = NaturalLanguageGenerator(language)

    def explain_single(
        self,
        grid: torch.Tensor,
        scenario: torch.Tensor,
        valid_positions: Optional[List[Tuple[int, int]]] = None,
    ) -> ExplanationReport:
        """
        Generate explanation for a single configuration.

        Args:
            grid: Configuration grid (5, H, W)
            scenario: Scenario parameters (4,)
            valid_positions: Valid positions for recommendations

        Returns:
            ExplanationReport
        """
        grid = grid.to(self.device)
        scenario = scenario.to(self.device)

        # Score
        with torch.no_grad():
            score = self.model.score_single(
                grid.unsqueeze(0),
                scenario.unsqueeze(0),
            ).item()

        # Feature importance
        feature_importance = self.attributor.get_feature_importance(
            grid.cpu(), scenario.cpu()
        )

        # Spatial attention
        spatial_attention = self.attention_analyzer.analyze_attention(
            grid.cpu(), scenario.cpu()
        )

        # Recommendations
        if valid_positions:
            recommendations = self.counterfactual.generate_recommendations(
                grid, scenario, valid_positions
            )
        else:
            recommendations = []

        # Confidence (placeholder - would come from uncertainty module)
        confidence = 0.85

        report = ExplanationReport(
            explanation_type=ExplanationType.SINGLE_CONFIG,
            summary="",
            score=score,
            confidence=confidence,
            spatial_attention=spatial_attention,
            feature_importance=feature_importance,
            recommendations=recommendations,
            detailed_analysis={
                'score': score,
                'n_hotspots': len(spatial_attention.hotspots),
                'n_bottlenecks': len(spatial_attention.bottlenecks),
            },
        )

        # Generate summary
        report.summary = self.nlg.generate_summary(report)

        return report

    def explain_comparison(
        self,
        grid_a: torch.Tensor,
        scenario_a: torch.Tensor,
        grid_b: torch.Tensor,
        scenario_b: torch.Tensor,
    ) -> ExplanationReport:
        """
        Generate explanation for a pairwise comparison.

        Args:
            grid_a, scenario_a: Configuration A
            grid_b, scenario_b: Configuration B

        Returns:
            ExplanationReport
        """
        grid_a = grid_a.to(self.device)
        scenario_a = scenario_a.to(self.device)
        grid_b = grid_b.to(self.device)
        scenario_b = scenario_b.to(self.device)

        # Get comparison result
        with torch.no_grad():
            outputs = self.model(
                grid_a.unsqueeze(0),
                scenario_a.unsqueeze(0),
                grid_b.unsqueeze(0),
                scenario_b.unsqueeze(0),
            )
            logit = outputs['logit'].item()
            prob = torch.sigmoid(outputs['logit']).item()

        # Analyze both configurations
        attention_a = self.attention_analyzer.analyze_attention(grid_a.cpu(), scenario_a.cpu())
        attention_b = self.attention_analyzer.analyze_attention(grid_b.cpu(), scenario_b.cpu())

        importance_a = self.attributor.get_feature_importance(grid_a.cpu(), scenario_a.cpu())
        importance_b = self.attributor.get_feature_importance(grid_b.cpu(), scenario_b.cpu())

        # Determine winner and confidence
        if prob > 0.5:
            winner = "A"
            confidence = prob
        else:
            winner = "B"
            confidence = 1 - prob

        report = ExplanationReport(
            explanation_type=ExplanationType.RANKING_COMPARISON,
            summary="",
            score=logit,
            confidence=confidence,
            spatial_attention=attention_a if winner == "A" else attention_b,
            feature_importance=importance_a if winner == "A" else importance_b,
            recommendations=[],
            detailed_analysis={
                'winner': winner,
                'logit': logit,
                'probability_a_wins': prob,
                'score_a': outputs['score_a'].item(),
                'score_b': outputs['score_b'].item(),
                'config_a_hotspots': len(attention_a.hotspots),
                'config_b_hotspots': len(attention_b.hotspots),
            },
        )

        report.summary = self.nlg.generate_summary(report)

        return report

    def export_report(
        self,
        report: ExplanationReport,
        format: str = "json",
    ) -> str:
        """
        Export report to specified format.

        Args:
            report: ExplanationReport to export
            format: Output format ("json", "markdown", "html")

        Returns:
            Formatted report string
        """
        if format == "json":
            return self._to_json(report)
        elif format == "markdown":
            return self._to_markdown(report)
        elif format == "html":
            return self._to_html(report)
        else:
            raise ValueError(f"Unknown format: {format}")

    def _to_json(self, report: ExplanationReport) -> str:
        """Convert report to JSON."""
        data = {
            'explanation_type': report.explanation_type.value,
            'summary': report.summary,
            'score': report.score,
            'confidence': report.confidence,
            'recommendations': [
                {
                    'action': r.action,
                    'element_type': r.element_type,
                    'current_position': r.current_position,
                    'suggested_position': r.suggested_position,
                    'expected_improvement': r.expected_improvement,
                    'reasoning': r.reasoning,
                }
                for r in report.recommendations
            ],
            'detailed_analysis': report.detailed_analysis,
        }
        return json.dumps(data, indent=2, ensure_ascii=False)

    def _to_markdown(self, report: ExplanationReport) -> str:
        """Convert report to Markdown."""
        lines = [
            f"# Configuration Analysis Report",
            "",
            f"## Summary",
            f"{report.summary}",
            "",
            f"**Score:** {report.score:.4f}",
            f"**Confidence:** {report.confidence:.1%}",
            "",
        ]

        if report.recommendations:
            lines.extend([
                "## Recommendations",
                "",
            ])
            for i, rec in enumerate(report.recommendations, 1):
                lines.append(f"{i}. **{rec.action.title()} {rec.element_type}**")
                lines.append(f"   - {rec.reasoning}")
                lines.append(f"   - Expected improvement: {rec.expected_improvement:.3f}")
                lines.append("")

        if report.spatial_attention and report.spatial_attention.bottlenecks:
            lines.extend([
                "## Identified Bottlenecks",
                "",
            ])
            for bn in report.spatial_attention.bottlenecks:
                lines.append(f"- **{bn['description']}** (severity: {bn['severity']:.2f})")
            lines.append("")

        return "\n".join(lines)

    def _to_html(self, report: ExplanationReport) -> str:
        """Convert report to HTML."""
        html = f"""
        <html>
        <head>
            <title>Configuration Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background: #f0f0f0; padding: 15px; border-radius: 5px; }}
                .metric {{ display: inline-block; margin-right: 20px; }}
                .recommendations {{ margin-top: 20px; }}
                .recommendation {{ border-left: 3px solid #4CAF50; padding-left: 10px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <h1>Configuration Analysis Report</h1>
            <div class="summary">
                <p>{report.summary}</p>
            </div>
            <div class="metrics">
                <div class="metric"><strong>Score:</strong> {report.score:.4f}</div>
                <div class="metric"><strong>Confidence:</strong> {report.confidence:.1%}</div>
            </div>
        """

        if report.recommendations:
            html += "<div class='recommendations'><h2>Recommendations</h2>"
            for rec in report.recommendations:
                html += f"""
                <div class="recommendation">
                    <strong>{rec.action.title()} {rec.element_type}</strong>
                    <p>{rec.reasoning}</p>
                    <small>Expected improvement: {rec.expected_improvement:.3f}</small>
                </div>
                """
            html += "</div>"

        html += "</body></html>"
        return html

"""
Interactive Pairwise Comparison Visualization

Visualizes how the ranking model compares two floor plans side-by-side,
showing the complete processing pipeline and comparison results.

Usage:
    # Compare two specific floor plans
    python -m ml.ranking_v2.visualize_comparison \
        --checkpoint checkpoints/ranking_v2/best_model.pt \
        --plan-a floor_plans/plan_001.npz \
        --plan-b floor_plans/plan_002.npz \
        --output comparison.html

    # Compare random pair from dataset
    python -m ml.ranking_v2.visualize_comparison \
        --checkpoint checkpoints/ranking_v2/best_model.pt \
        --random-pair \
        --data-dir combined_fast \
        --output comparison.html
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from .config import RankingV2Config
from .model import CrossAttentionRanker
from .train import load_checkpoint
from .visualize import GradCAM, encode_grid_5ch


def load_floor_plan_npz(
    npz_path: str,
    target_size: Tuple[int, int] = (96, 128)
) -> torch.Tensor:
    """
    Load floor plan from NPZ file and encode to 5-channel tensor.

    Args:
        npz_path: Path to NPZ file containing 'grid' array
        target_size: Target (height, width) for output tensor

    Returns:
        Tensor of shape (5, H, W)
    """
    data = np.load(npz_path)
    grid_2d = data['grid']
    encoded = encode_grid_5ch(grid_2d, target_size)
    return torch.from_numpy(encoded)


def create_floor_plan_display(
    grid: torch.Tensor,
    title: str,
    score: float
) -> go.Heatmap:
    """
    Create floor plan visualization as Plotly heatmap.

    Uses channel 1 (passable areas) combined with walls for display.
    """
    # Combine channels for visualization
    # Channel 0: walls, Channel 1: passable, Channel 4: valid mask
    walls = grid[0].cpu().numpy()
    passable = grid[1].cpu().numpy()
    valid = grid[4].cpu().numpy()

    # Create combined display: -1 for walls, 1 for passable, 0 for invalid
    display = np.zeros_like(passable)
    display[walls > 0.5] = -1  # Walls
    display[passable > 0.5] = 1  # Passable
    display[valid < 0.5] = 0  # Invalid/padding

    return go.Heatmap(
        z=display,
        colorscale=[
            [0, 'rgb(50, 50, 50)'],      # Invalid - dark gray
            [0.5, 'rgb(100, 100, 100)'],  # Walls - gray
            [1, 'rgb(255, 255, 255)']     # Passable - white
        ],
        showscale=False,
        hovertemplate='Row: %{y}<br>Col: %{x}<br>Value: %{z}<extra></extra>'
    )


def create_gradcam_display(
    grid: torch.Tensor,
    cam: np.ndarray,
    title: str
) -> go.Heatmap:
    """
    Create GradCAM overlay visualization.

    Shows attention intensity as color overlay on floor plan.
    """
    # Mask CAM to valid regions
    valid = grid[4].cpu().numpy()
    cam_masked = cam.copy()
    cam_masked[valid < 0.5] = np.nan

    return go.Heatmap(
        z=cam_masked,
        colorscale='Hot',
        showscale=True,
        colorbar=dict(title='Attention', len=0.3),
        hovertemplate='Row: %{y}<br>Col: %{x}<br>Attention: %{z:.3f}<extra></extra>'
    )


def create_comparison_figure(
    model: CrossAttentionRanker,
    grid_a: torch.Tensor,
    scenario_a: torch.Tensor,
    grid_b: torch.Tensor,
    scenario_b: torch.Tensor,
    device: torch.device,
    plan_a_name: str = "Plan A",
    plan_b_name: str = "Plan B"
) -> go.Figure:
    """
    Create complete interactive comparison visualization.

    Args:
        model: Trained ranking model
        grid_a, grid_b: Floor plan tensors (5, H, W)
        scenario_a, scenario_b: Scenario parameter tensors (4,)
        device: Computation device
        plan_a_name, plan_b_name: Display names for plans

    Returns:
        Plotly Figure with complete visualization
    """
    model.eval()

    # Move to device and add batch dimension
    grid_a = grid_a.to(device).unsqueeze(0)
    grid_b = grid_b.to(device).unsqueeze(0)
    scenario_a = scenario_a.to(device).unsqueeze(0)
    scenario_b = scenario_b.to(device).unsqueeze(0)

    # Run model forward pass
    with torch.no_grad():
        outputs = model(grid_a, scenario_a, grid_b, scenario_b)

    score_a = outputs['score_a'].item()
    score_b = outputs['score_b'].item()
    logit = outputs['logit'].item()
    aux_a = {k: v.item() for k, v in outputs['aux_a'].items()}
    aux_b = {k: v.item() for k, v in outputs['aux_b'].items()}

    # Compute GradCAM for both plans
    gradcam = GradCAM(model)
    cam_a = gradcam.generate(grid_a.squeeze(0), scenario_a.squeeze(0))
    cam_b = gradcam.generate(grid_b.squeeze(0), scenario_b.squeeze(0))

    # Compute probability
    prob_a_wins = 1 / (1 + np.exp(-logit))  # Sigmoid

    # Determine winner
    if logit > 0:
        winner_text = f"{plan_a_name} is predicted better"
        winner_color = "green"
    else:
        winner_text = f"{plan_b_name} is predicted better"
        winner_color = "blue"

    # Create figure with subplots
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            f'{plan_a_name} (Score: {score_a:.3f})',
            f'{plan_b_name} (Score: {score_b:.3f})',
            f'GradCAM Attention - {plan_a_name}',
            f'GradCAM Attention - {plan_b_name}',
            '', ''
        ),
        row_heights=[0.35, 0.35, 0.15, 0.15],
        vertical_spacing=0.08,
        horizontal_spacing=0.1,
        specs=[
            [{"type": "heatmap"}, {"type": "heatmap"}],
            [{"type": "heatmap"}, {"type": "heatmap"}],
            [{"type": "bar", "colspan": 2}, None],
            [{"type": "bar", "colspan": 2}, None]
        ]
    )

    # Row 1: Floor Plans
    fig.add_trace(
        create_floor_plan_display(grid_a.squeeze(0), plan_a_name, score_a),
        row=1, col=1
    )
    fig.add_trace(
        create_floor_plan_display(grid_b.squeeze(0), plan_b_name, score_b),
        row=1, col=2
    )

    # Row 2: GradCAM Attention
    fig.add_trace(
        create_gradcam_display(grid_a.squeeze(0), cam_a, plan_a_name),
        row=2, col=1
    )
    fig.add_trace(
        create_gradcam_display(grid_b.squeeze(0), cam_b, plan_b_name),
        row=2, col=2
    )

    # Row 3: Score Comparison Bar
    fig.add_trace(
        go.Bar(
            x=[score_a, score_b],
            y=[plan_a_name, plan_b_name],
            orientation='h',
            marker_color=['green', 'blue'],
            text=[f'{score_a:.3f}', f'{score_b:.3f}'],
            textposition='outside',
            hovertemplate='%{y}: %{x:.4f}<extra></extra>'
        ),
        row=3, col=1
    )

    # Row 4: Auxiliary Predictions (if available)
    if aux_a and aux_b:
        tasks = list(aux_a.keys())
        x_labels = []
        values_a = []
        values_b = []

        for task in tasks:
            x_labels.append(task.replace('_', ' ').title())
            values_a.append(aux_a[task])
            values_b.append(aux_b[task])

        fig.add_trace(
            go.Bar(
                name=plan_a_name,
                x=x_labels,
                y=values_a,
                marker_color='green',
                text=[f'{v:.3f}' for v in values_a],
                textposition='outside'
            ),
            row=4, col=1
        )
        fig.add_trace(
            go.Bar(
                name=plan_b_name,
                x=x_labels,
                y=values_b,
                marker_color='blue',
                text=[f'{v:.3f}' for v in values_b],
                textposition='outside'
            ),
            row=4, col=1
        )

    # Format scenario for display
    scenario_vals = scenario_a.squeeze(0).cpu().numpy()
    scenario_text = (
        f"Agents: {scenario_vals[0]:.0f}, "
        f"Fires: {scenario_vals[1]:.0f}, "
        f"Spread Rate: {scenario_vals[2]:.2f}, "
        f"Discovery Delay: {scenario_vals[3]:.0f}"
    )

    # Update layout
    fig.update_layout(
        title=dict(
            text=(
                f"<b>Pairwise Floor Plan Comparison</b><br>"
                f"<span style='font-size:14px'>Scenario: {scenario_text}</span><br>"
                f"<span style='font-size:16px; color:{winner_color}'>"
                f"{winner_text} | Logit: {logit:+.3f} | "
                f"P({plan_a_name} wins): {prob_a_wins:.1%}</span>"
            ),
            x=0.5,
            xanchor='center'
        ),
        height=1200,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.05,
            xanchor="center",
            x=0.5
        )
    )

    # Update axes to maintain aspect ratio for floor plans
    for row in [1, 2]:
        for col in [1, 2]:
            fig.update_xaxes(scaleanchor=f"y{row if row == 1 else row + 1}", row=row, col=col)
            fig.update_yaxes(autorange='reversed', row=row, col=col)

    # Update bar chart axes
    fig.update_xaxes(title_text="Score", row=3, col=1)
    fig.update_yaxes(title_text="", row=3, col=1)

    if aux_a:
        fig.update_xaxes(title_text="Metric", row=4, col=1)
        fig.update_yaxes(title_text="Value", row=4, col=1)
        fig.update_layout(barmode='group')

    return fig


def visualize_pairwise_comparison(
    checkpoint_path: str,
    plan_a_path: Optional[str] = None,
    plan_b_path: Optional[str] = None,
    output_path: str = "comparison.html",
    data_dir: Optional[str] = None,
    random_pair: bool = False,
    scenario: Optional[Dict] = None,
    device: Optional[torch.device] = None
) -> str:
    """
    Generate interactive comparison visualization.

    Args:
        checkpoint_path: Path to model checkpoint
        plan_a_path: Path to first floor plan NPZ
        plan_b_path: Path to second floor plan NPZ
        output_path: Output HTML file path
        data_dir: Data directory (for random pair mode)
        random_pair: If True, pick random pair from data_dir
        scenario: Scenario parameters dict, or None for defaults
        device: Computation device

    Returns:
        Path to generated HTML file
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "plotly is required for interactive visualization. "
            "Install with: pip install plotly"
        )

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    print(f"Loading model from {checkpoint_path}...")
    model, checkpoint = load_checkpoint(checkpoint_path, device=device)
    config = RankingV2Config(**checkpoint['config'])
    target_size = config.target_grid_size

    # Get floor plans
    if random_pair:
        if data_dir is None:
            data_dir = config.floor_plans_dir or "combined_fast/floor_plans"

        floor_plans_dir = Path(data_dir)
        if not floor_plans_dir.exists():
            # Try as subdirectory
            floor_plans_dir = Path(data_dir) / "floor_plans"

        npz_files = sorted(floor_plans_dir.glob("*.npz"))
        if len(npz_files) < 2:
            raise ValueError(f"Need at least 2 floor plans in {floor_plans_dir}")

        # Pick random pair
        indices = np.random.choice(len(npz_files), 2, replace=False)
        plan_a_path = str(npz_files[indices[0]])
        plan_b_path = str(npz_files[indices[1]])
        print(f"Selected random pair:")
        print(f"  Plan A: {Path(plan_a_path).name}")
        print(f"  Plan B: {Path(plan_b_path).name}")
    else:
        if plan_a_path is None or plan_b_path is None:
            raise ValueError(
                "Must provide --plan-a and --plan-b, or use --random-pair"
            )

    # Load floor plans
    print("Loading floor plans...")
    grid_a = load_floor_plan_npz(plan_a_path, target_size)
    grid_b = load_floor_plan_npz(plan_b_path, target_size)

    # Create scenario tensor
    if scenario is None:
        # Default scenario
        scenario = {
            'agent_count': 50,
            'num_fires': 3,
            'fire_spread_rate': 0.1,
            'fire_discovery_delay': 10
        }

    scenario_tensor = torch.tensor([
        scenario.get('agent_count', 50),
        scenario.get('num_fires', 3),
        scenario.get('fire_spread_rate', 0.1),
        scenario.get('fire_discovery_delay', 10)
    ], dtype=torch.float32)

    # Create visualization
    print("Generating visualization...")
    plan_a_name = Path(plan_a_path).stem
    plan_b_name = Path(plan_b_path).stem

    fig = create_comparison_figure(
        model=model,
        grid_a=grid_a,
        scenario_a=scenario_tensor,
        grid_b=grid_b,
        scenario_b=scenario_tensor,
        device=device,
        plan_a_name=plan_a_name,
        plan_b_name=plan_b_name
    )

    # Save to HTML
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs=True, full_html=True)

    print(f"Saved interactive visualization to {output_path}")
    return str(output_path)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Interactive pairwise floor plan comparison visualization"
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)"
    )

    parser.add_argument(
        '--plan-a',
        type=str,
        default=None,
        help="Path to first floor plan NPZ file"
    )

    parser.add_argument(
        '--plan-b',
        type=str,
        default=None,
        help="Path to second floor plan NPZ file"
    )

    parser.add_argument(
        '--random-pair',
        action='store_true',
        help="Pick random pair from data directory"
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help="Data directory containing floor_plans/ (for --random-pair)"
    )

    parser.add_argument(
        '--scenario',
        type=str,
        default=None,
        help="Scenario parameters as JSON string, e.g., "
             "'{\"agent_count\": 50, \"num_fires\": 3}'"
    )

    parser.add_argument(
        '--output',
        type=str,
        default='comparison.html',
        help="Output HTML file path (default: comparison.html)"
    )

    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help="Device to use (default: auto-detect)"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Parse scenario if provided
    scenario = None
    if args.scenario:
        scenario = json.loads(args.scenario)

    # Parse device
    device = None
    if args.device:
        device = torch.device(args.device)

    # Generate visualization
    visualize_pairwise_comparison(
        checkpoint_path=args.checkpoint,
        plan_a_path=args.plan_a,
        plan_b_path=args.plan_b,
        output_path=args.output,
        data_dir=args.data_dir,
        random_pair=args.random_pair,
        scenario=scenario,
        device=device
    )


if __name__ == '__main__':
    main()

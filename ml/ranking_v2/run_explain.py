"""
CLI Entry Point for Explainable Report Generation

Generate human-readable analysis reports for floor plan configurations.

Usage:
    # Analyze a single configuration
    python -m ml.ranking_v2.run_explain --checkpoint model.pt --input data.npz

    # Compare two configurations
    python -m ml.ranking_v2.run_explain --checkpoint model.pt --input-a config_a.npz --input-b config_b.npz

    # Specify output format and language
    python -m ml.ranking_v2.run_explain --checkpoint model.pt --input data.npz \
        --format html --language zh --output report.html

    # Batch analysis
    python -m ml.ranking_v2.run_explain --checkpoint model.pt --input-dir configs/ \
        --output-dir reports/ --format markdown

Examples:
    # Quick analysis with Chinese report
    python -m ml.ranking_v2.run_explain -c best_model.pt -i floor_plan.npz -l zh

    # Generate HTML report with visualization
    python -m ml.ranking_v2.run_explain -c best_model.pt -i floor_plan.npz -f html --save-heatmap
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import torch

from .config import RankingV2Config
from .model import CrossAttentionRanker
from .explainer import (
    ExplanationPipeline,
    ExplanationReport,
    ExplanationType,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate explainable reports for floor plan configurations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required arguments
    parser.add_argument(
        '-c', '--checkpoint',
        type=str,
        required=True,
        help='Path to trained model checkpoint (.pt file)'
    )

    # Input modes (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '-i', '--input',
        type=str,
        help='Path to input configuration (.npz file with grid and scenario)'
    )
    input_group.add_argument(
        '--input-a',
        type=str,
        help='Path to first configuration for comparison'
    )
    input_group.add_argument(
        '--input-dir',
        type=str,
        help='Directory containing multiple configurations for batch analysis'
    )

    # For comparison mode
    parser.add_argument(
        '--input-b',
        type=str,
        help='Path to second configuration for comparison (requires --input-a)'
    )

    # Output options
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output file path (default: stdout for json, auto-named for html/markdown)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for batch analysis'
    )
    parser.add_argument(
        '-f', '--format',
        type=str,
        choices=['json', 'markdown', 'html', 'text'],
        default='text',
        help='Output format (default: text)'
    )
    parser.add_argument(
        '-l', '--language',
        type=str,
        choices=['en', 'zh'],
        default='zh',
        help='Report language (default: zh)'
    )

    # Visualization options
    parser.add_argument(
        '--save-heatmap',
        action='store_true',
        help='Save attention heatmap as PNG'
    )
    parser.add_argument(
        '--heatmap-path',
        type=str,
        help='Path for heatmap image (default: <input>_heatmap.png)'
    )

    # Model options
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device to use: cuda, cpu, or auto (default: auto)'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to model config YAML (if not embedded in checkpoint)'
    )

    # Analysis options
    parser.add_argument(
        '--top-k-recommendations',
        type=int,
        default=5,
        help='Number of improvement recommendations to generate (default: 5)'
    )
    parser.add_argument(
        '--no-recommendations',
        action='store_true',
        help='Skip generating improvement recommendations (faster)'
    )

    # Scenario override
    parser.add_argument(
        '--agents',
        type=int,
        help='Override agent count in scenario'
    )
    parser.add_argument(
        '--fires',
        type=int,
        help='Override number of fires in scenario'
    )
    parser.add_argument(
        '--spread-rate',
        type=float,
        help='Override fire spread rate in scenario'
    )
    parser.add_argument(
        '--delay',
        type=float,
        help='Override fire discovery delay in scenario'
    )

    return parser.parse_args()


def load_model(
    checkpoint_path: str,
    config_path: Optional[str] = None,
    device: str = 'auto'
) -> Tuple[CrossAttentionRanker, RankingV2Config, torch.device]:
    """
    Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        config_path: Optional path to config YAML
        device: Device specification

    Returns:
        Tuple of (model, config, device)
    """
    # Determine device
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    print(f"Loading model from: {checkpoint_path}", file=sys.stderr)
    print(f"Using device: {device}", file=sys.stderr)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Get config
    if 'config' in checkpoint:
        config = checkpoint['config']
    elif config_path:
        import yaml
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = RankingV2Config(**config_dict)
    else:
        print("Warning: No config found, using defaults", file=sys.stderr)
        config = RankingV2Config()

    # Build model
    model = CrossAttentionRanker(config)

    # Load weights
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"Model loaded successfully", file=sys.stderr)

    return model, config, device


def load_input(
    input_path: str,
    agents: Optional[int] = None,
    fires: Optional[int] = None,
    spread_rate: Optional[float] = None,
    delay: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
    """
    Load input configuration from file.

    Args:
        input_path: Path to .npz file
        agents, fires, spread_rate, delay: Optional scenario overrides

    Returns:
        Tuple of (grid, scenario, valid_positions)
    """
    data = np.load(input_path)

    # Load grid
    if 'grid' in data:
        grid = torch.from_numpy(data['grid']).float()
    elif 'floor_plan' in data:
        grid = torch.from_numpy(data['floor_plan']).float()
    else:
        raise ValueError(f"No 'grid' or 'floor_plan' found in {input_path}")

    # Load scenario
    if 'scenario' in data:
        scenario = torch.from_numpy(data['scenario']).float()
    else:
        # Default scenario
        scenario = torch.tensor([100.0, 1.0, 0.5, 30.0])

    # Apply overrides
    if agents is not None:
        scenario[0] = float(agents)
    if fires is not None:
        scenario[1] = float(fires)
    if spread_rate is not None:
        scenario[2] = float(spread_rate)
    if delay is not None:
        scenario[3] = float(delay)

    # Find valid positions (passable areas)
    if 'valid_positions' in data:
        valid_positions = [tuple(p) for p in data['valid_positions']]
    else:
        passable = grid[1].numpy() > 0.5
        valid_positions = [
            (int(y), int(x))
            for y in range(grid.shape[1])
            for x in range(grid.shape[2])
            if passable[y, x]
        ]

    return grid, scenario, valid_positions


def format_text_report(report: ExplanationReport, language: str = 'zh') -> str:
    """Format report as plain text."""
    lines = []

    if language == 'zh':
        lines.append("=" * 60)
        lines.append("  配置分析報告")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"摘要: {report.summary}")
        lines.append("")
        lines.append(f"評分: {report.score:.4f}")
        lines.append(f"信心度: {report.confidence:.1%}")
        lines.append("")

        if report.spatial_attention and report.spatial_attention.bottlenecks:
            lines.append("識別到的瓶頸:")
            for bn in report.spatial_attention.bottlenecks:
                lines.append(f"  - {bn['description']} (嚴重度: {bn['severity']:.2f})")
            lines.append("")

        if report.spatial_attention and report.spatial_attention.hotspots:
            lines.append("關注熱點:")
            for hs in report.spatial_attention.hotspots[:3]:
                lines.append(f"  - {hs['description']} (重要性: {hs['importance']:.2f})")
            lines.append("")

        if report.feature_importance:
            lines.append("特徵重要性:")
            for name, value in report.feature_importance.scenario_importance.items():
                name_zh = {
                    'agent_count': '人員數量',
                    'num_fires': '火源數量',
                    'fire_spread_rate': '火勢擴散速率',
                    'fire_discovery_delay': '發現延遲'
                }.get(name, name)
                lines.append(f"  - {name_zh}: {value:.4f}")
            lines.append("")

        if report.recommendations:
            lines.append("改善建議:")
            for i, rec in enumerate(report.recommendations, 1):
                lines.append(f"  {i}. {rec.reasoning}")
                lines.append(f"     預期改善: {rec.expected_improvement:.4f}")
            lines.append("")

    else:  # English
        lines.append("=" * 60)
        lines.append("  Configuration Analysis Report")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"Summary: {report.summary}")
        lines.append("")
        lines.append(f"Score: {report.score:.4f}")
        lines.append(f"Confidence: {report.confidence:.1%}")
        lines.append("")

        if report.spatial_attention and report.spatial_attention.bottlenecks:
            lines.append("Identified Bottlenecks:")
            for bn in report.spatial_attention.bottlenecks:
                lines.append(f"  - {bn['description']} (severity: {bn['severity']:.2f})")
            lines.append("")

        if report.recommendations:
            lines.append("Recommendations:")
            for i, rec in enumerate(report.recommendations, 1):
                lines.append(f"  {i}. {rec.reasoning}")
                lines.append(f"     Expected improvement: {rec.expected_improvement:.4f}")
            lines.append("")

    return "\n".join(lines)


def save_heatmap(
    report: ExplanationReport,
    output_path: str,
    grid: Optional[torch.Tensor] = None
):
    """Save attention heatmap as image."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, skipping heatmap", file=sys.stderr)
        return

    if report.spatial_attention is None:
        print("Warning: No spatial attention data, skipping heatmap", file=sys.stderr)
        return

    heatmap = report.spatial_attention.heatmap

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Show heatmap
    im = ax.imshow(heatmap, cmap='hot', interpolation='bilinear')
    plt.colorbar(im, ax=ax, label='Attention')

    # Overlay grid structure if available
    if grid is not None:
        walls = grid[0].numpy()
        ax.contour(walls, levels=[0.5], colors='white', linewidths=0.5)

    # Mark bottlenecks
    for bn in report.spatial_attention.bottlenecks:
        y, x = bn['position']
        ax.plot(x, y, 'bx', markersize=15, markeredgewidth=3)
        ax.annotate(
            'Bottleneck',
            (x, y),
            xytext=(10, 10),
            textcoords='offset points',
            color='white',
            fontsize=8
        )

    # Mark hotspots
    for hs in report.spatial_attention.hotspots[:3]:
        y, x = hs['position']
        ax.plot(x, y, 'go', markersize=10, markeredgewidth=2, fillstyle='none')

    ax.set_title('Attention Heatmap with Bottlenecks')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Heatmap saved to: {output_path}", file=sys.stderr)


def analyze_single(
    pipeline: ExplanationPipeline,
    grid: torch.Tensor,
    scenario: torch.Tensor,
    valid_positions: List[Tuple[int, int]],
    args: argparse.Namespace
) -> ExplanationReport:
    """Analyze a single configuration."""
    if args.no_recommendations:
        valid_positions = None

    report = pipeline.explain_single(grid, scenario, valid_positions)

    # Limit recommendations
    if report.recommendations:
        report.recommendations = report.recommendations[:args.top_k_recommendations]

    return report


def analyze_comparison(
    pipeline: ExplanationPipeline,
    grid_a: torch.Tensor,
    scenario_a: torch.Tensor,
    grid_b: torch.Tensor,
    scenario_b: torch.Tensor,
) -> ExplanationReport:
    """Analyze comparison between two configurations."""
    return pipeline.explain_comparison(grid_a, scenario_a, grid_b, scenario_b)


def output_report(
    report: ExplanationReport,
    args: argparse.Namespace,
    grid: Optional[torch.Tensor] = None,
    input_path: Optional[str] = None
):
    """Output report in specified format."""
    # Generate content
    if args.format == 'text':
        content = format_text_report(report, args.language)
    elif args.format == 'json':
        from .explainer import ExplanationPipeline
        # Create temporary pipeline just for export
        content = json.dumps({
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
            'bottlenecks': [
                {
                    'position': bn['position'],
                    'severity': bn['severity'],
                    'description': bn['description'],
                }
                for bn in (report.spatial_attention.bottlenecks if report.spatial_attention else [])
            ],
        }, indent=2, ensure_ascii=False)
    elif args.format == 'markdown':
        content = _to_markdown(report)
    elif args.format == 'html':
        content = _to_html(report, args.language)
    else:
        raise ValueError(f"Unknown format: {args.format}")

    # Output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Report saved to: {output_path}", file=sys.stderr)
    else:
        print(content)

    # Save heatmap if requested
    if args.save_heatmap:
        if args.heatmap_path:
            heatmap_path = args.heatmap_path
        elif input_path:
            heatmap_path = str(Path(input_path).with_suffix('')) + '_heatmap.png'
        elif args.output:
            heatmap_path = str(Path(args.output).with_suffix('')) + '_heatmap.png'
        else:
            heatmap_path = 'heatmap.png'

        save_heatmap(report, heatmap_path, grid)


def _to_markdown(report: ExplanationReport) -> str:
    """Convert report to Markdown."""
    lines = [
        "# Configuration Analysis Report",
        "",
        "## Summary",
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


def _to_html(report: ExplanationReport, language: str = 'zh') -> str:
    """Convert report to HTML."""
    title = "配置分析報告" if language == 'zh' else "Configuration Analysis Report"
    rec_title = "改善建議" if language == 'zh' else "Recommendations"
    bn_title = "識別到的瓶頸" if language == 'zh' else "Identified Bottlenecks"

    html = f"""<!DOCTYPE html>
<html lang="{language}">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 40px;
            background: #f5f5f5;
        }}
        .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        .summary {{ background: #e8f5e9; padding: 20px; border-radius: 5px; margin: 20px 0; }}
        .metrics {{ display: flex; gap: 30px; margin: 20px 0; }}
        .metric {{ background: #f5f5f5; padding: 15px 25px; border-radius: 5px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #4CAF50; }}
        .metric-label {{ color: #666; font-size: 14px; }}
        .recommendations {{ margin-top: 30px; }}
        .recommendation {{
            border-left: 4px solid #4CAF50;
            padding: 15px 20px;
            margin: 15px 0;
            background: #fafafa;
            border-radius: 0 5px 5px 0;
        }}
        .recommendation strong {{ color: #2e7d32; }}
        .bottlenecks {{ margin-top: 30px; }}
        .bottleneck {{
            border-left: 4px solid #ff9800;
            padding: 10px 15px;
            margin: 10px 0;
            background: #fff3e0;
            border-radius: 0 5px 5px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <div class="summary">
            <p>{report.summary}</p>
        </div>
        <div class="metrics">
            <div class="metric">
                <div class="metric-value">{report.score:.4f}</div>
                <div class="metric-label">{"評分" if language == 'zh' else "Score"}</div>
            </div>
            <div class="metric">
                <div class="metric-value">{report.confidence:.1%}</div>
                <div class="metric-label">{"信心度" if language == 'zh' else "Confidence"}</div>
            </div>
        </div>
"""

    if report.recommendations:
        html += f'<div class="recommendations"><h2>{rec_title}</h2>'
        for rec in report.recommendations:
            html += f"""
            <div class="recommendation">
                <strong>{rec.action.title()} {rec.element_type}</strong>
                <p>{rec.reasoning}</p>
                <small>{"預期改善" if language == 'zh' else "Expected improvement"}: {rec.expected_improvement:.3f}</small>
            </div>
            """
        html += '</div>'

    if report.spatial_attention and report.spatial_attention.bottlenecks:
        html += f'<div class="bottlenecks"><h2>{bn_title}</h2>'
        for bn in report.spatial_attention.bottlenecks:
            html += f"""
            <div class="bottleneck">
                <strong>{bn['description']}</strong>
                <span style="float:right">{"嚴重度" if language == 'zh' else "Severity"}: {bn['severity']:.2f}</span>
            </div>
            """
        html += '</div>'

    html += """
    </div>
</body>
</html>"""

    return html


def main():
    """Main entry point."""
    args = parse_args()

    # Validate arguments
    if args.input_a and not args.input_b:
        print("Error: --input-b is required when using --input-a", file=sys.stderr)
        sys.exit(1)

    # Load model
    model, config, device = load_model(args.checkpoint, args.config, args.device)

    # Create pipeline
    pipeline = ExplanationPipeline(
        model=model,
        config=config,
        device=device,
        language=args.language
    )

    # Process based on mode
    if args.input:
        # Single configuration analysis
        grid, scenario, valid_positions = load_input(
            args.input,
            args.agents, args.fires, args.spread_rate, args.delay
        )

        print(f"Analyzing: {args.input}", file=sys.stderr)
        print(f"Grid shape: {tuple(grid.shape)}", file=sys.stderr)
        print(f"Scenario: agents={scenario[0]:.0f}, fires={scenario[1]:.0f}, "
              f"spread={scenario[2]:.2f}, delay={scenario[3]:.0f}s", file=sys.stderr)

        report = analyze_single(pipeline, grid, scenario, valid_positions, args)
        output_report(report, args, grid, args.input)

    elif args.input_a:
        # Comparison analysis
        grid_a, scenario_a, _ = load_input(
            args.input_a,
            args.agents, args.fires, args.spread_rate, args.delay
        )
        grid_b, scenario_b, _ = load_input(
            args.input_b,
            args.agents, args.fires, args.spread_rate, args.delay
        )

        print(f"Comparing: {args.input_a} vs {args.input_b}", file=sys.stderr)

        report = analyze_comparison(pipeline, grid_a, scenario_a, grid_b, scenario_b)
        output_report(report, args)

    elif args.input_dir:
        # Batch analysis
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir) if args.output_dir else input_dir / 'reports'
        output_dir.mkdir(parents=True, exist_ok=True)

        input_files = list(input_dir.glob('*.npz'))
        print(f"Found {len(input_files)} configurations to analyze", file=sys.stderr)

        for input_file in input_files:
            print(f"Processing: {input_file.name}", file=sys.stderr)

            grid, scenario, valid_positions = load_input(
                str(input_file),
                args.agents, args.fires, args.spread_rate, args.delay
            )

            report = analyze_single(pipeline, grid, scenario, valid_positions, args)

            # Determine output extension
            ext = {'json': '.json', 'markdown': '.md', 'html': '.html', 'text': '.txt'}
            output_file = output_dir / (input_file.stem + '_report' + ext[args.format])

            # Temporarily set output path
            original_output = args.output
            args.output = str(output_file)
            output_report(report, args, grid, str(input_file))
            args.output = original_output

        print(f"Batch analysis complete. Reports saved to: {output_dir}", file=sys.stderr)


if __name__ == '__main__':
    main()

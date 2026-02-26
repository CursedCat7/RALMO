"""RALMO CLI — Command-line interface for the orchestration engine.

Provides commands to run speculative inference, benchmark, and inspect
configuration. Built with Typer + Hydra for flexible parameter management.
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="ralmo",
    help="RALMO — Resource-Adaptive Language Model Orchestration",
    add_completion=False,
)
console = Console()


def _load_config(config_path: str | None = None) -> DictConfig:
    """Load Hydra configuration.

    Args:
        config_path: Optional path to a YAML config file.
                     Defaults to configs/default.yaml relative to project root.

    Returns:
        DictConfig with loaded configuration.
    """
    if config_path is not None:
        cfg_path = Path(config_path)
    else:
        # Search for default config relative to the CLI location
        cli_dir = Path(__file__).resolve().parent
        project_root = cli_dir.parent.parent
        cfg_path = project_root / "configs" / "default.yaml"

    if not cfg_path.exists():
        console.print(f"[red]Config file not found: {cfg_path}[/red]")
        raise typer.Exit(code=1)

    cfg = OmegaConf.load(str(cfg_path))
    assert isinstance(cfg, DictConfig)
    return cfg


@app.command()
def run(
    prompt: str = typer.Argument(..., help="Input prompt text for generation"),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to YAML config file"),
    max_tokens: int = typer.Option(512, "--max-tokens", "-m", help="Maximum output tokens"),
    draft_model: str | None = typer.Option(None, "--draft-model", help="Override draft model path"),
    target_model: str | None = typer.Option(
        None, "--target-model", help="Override target model path"
    ),
    tau: float | None = typer.Option(None, "--tau", help="Override acceptance threshold"),
    k: int | None = typer.Option(None, "-k", help="Override draft token count"),
    policy_type: str | None = typer.Option(
        None, "--policy-type", help="Override policy type (static/adaptive)"
    ),
    adaptive_alpha: float | None = typer.Option(None, "--alpha", help="Override adaptive alpha"),
    adaptive_tau0: float | None = typer.Option(None, "--tau0", help="Override adaptive base tau_0"),
    adaptive_h0: float | None = typer.Option(None, "--h0", help="Override adaptive h_0"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
) -> None:
    """Run speculative inference on a prompt.

    Loads the draft and target models, runs the speculative decoding engine,
    and outputs the generated text along with performance statistics.
    """
    # Set up logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )

    # Load config
    cfg = _load_config(config)

    # Apply CLI overrides
    if draft_model is not None:
        cfg.draft.model_path = draft_model
    if target_model is not None:
        cfg.target.model_path = target_model
    if tau is not None:
        cfg.speculative.tau = tau
    if k is not None:
        cfg.speculative.k = k
    if policy_type is not None:
        cfg.speculative.policy_type = policy_type

    if adaptive_alpha is not None or adaptive_tau0 is not None or adaptive_h0 is not None:
        if "adaptive" not in cfg.speculative:
            cfg.speculative.adaptive = {}
        if adaptive_alpha is not None:
            cfg.speculative.adaptive.alpha = adaptive_alpha
        if adaptive_tau0 is not None:
            cfg.speculative.adaptive.tau_0 = adaptive_tau0
        if adaptive_h0 is not None:
            cfg.speculative.adaptive.h_0 = adaptive_h0

    cfg.speculative.max_tokens = max_tokens
    cfg.logging.verbose = verbose

    console.print(
        Panel.fit(
            "[bold cyan]RALMO[/bold cyan] — Resource-Adaptive Language Model Orchestration",
            border_style="cyan",
        )
    )
    console.print(f"[dim]Draft: {cfg.draft.model_path}[/dim]")
    console.print(f"[dim]Target: {cfg.target.model_path}[/dim]")
    ptype = cfg.speculative.get("policy_type", "static")
    console.print(
        f"[dim]Policy={ptype}, τ={cfg.speculative.get('tau', -0.7)}, "
        f"k={cfg.speculative.k}, max_tokens={max_tokens}[/dim]"
    )
    console.print()

    # Initialize and run
    from ralmo_core.orchestrator import Orchestrator

    orch = Orchestrator(cfg)

    try:
        console.print("[yellow]Loading models...[/yellow]")
        orch.initialize()
        console.print("[green]Models loaded successfully.[/green]")
        console.print()

        console.print(f"[bold]Prompt:[/bold] {prompt}")
        console.print("[yellow]Generating...[/yellow]")
        console.print()

        result = orch.generate(prompt, max_tokens=max_tokens)

        # Display result
        console.print(Panel(result.text, title="Generated Output", border_style="green"))

        # Display stats
        stats_table = Table(title="Performance Statistics", border_style="blue")
        stats_table.add_column("Metric", style="bold")
        stats_table.add_column("Value", justify="right")

        stats = result.stats
        stats_table.add_row("Total Tokens", str(stats.total_tokens))
        stats_table.add_row("Draft Proposed", str(stats.draft_tokens_proposed))
        stats_table.add_row("Draft Accepted", str(stats.draft_tokens_accepted))
        stats_table.add_row("Draft Rejected", str(stats.draft_tokens_rejected))
        stats_table.add_row("Target Corrections", str(stats.target_corrections))
        stats_table.add_row("Acceptance Rate", f"{stats.acceptance_rate:.1%}")
        stats_table.add_row("Iterations", str(stats.iterations))
        stats_table.add_row("Latency", f"{stats.latency_ms:.1f} ms")
        stats_table.add_row("Energy", f"{result.energy_joules:.4f} J")
        stats_table.add_row("Finish Reason", result.finish_reason)

        console.print(stats_table)

    except FileNotFoundError as e:
        console.print(f"[red]Model file not found: {e}[/red]")
        console.print("[dim]Download GGUF model files and update config paths.[/dim]")
        raise typer.Exit(code=1) from e
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logging.exception("Orchestration error")
        raise typer.Exit(code=1) from e
    finally:
        orch.shutdown()


@app.command()
def config(
    config_path: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
) -> None:
    """Display the current configuration."""
    cfg = _load_config(config_path)
    console.print(
        Panel.fit(
            OmegaConf.to_yaml(cfg),
            title="RALMO Configuration",
            border_style="cyan",
        )
    )


@app.command()
def benchmark() -> None:
    """Run benchmark suite (placeholder for future implementation)."""
    console.print("[yellow]Benchmark mode is not yet implemented.[/yellow]")
    console.print("[dim]Coming in Phase 2: multi-dataset evaluation pipeline.[/dim]")
    raise typer.Exit(code=0)


if __name__ == "__main__":
    app()

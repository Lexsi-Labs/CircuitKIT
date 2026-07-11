"""
CircuitKit CLI - Main entry point for command-line interface.
"""

import os
from pathlib import Path

import click
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ..utils.device import get_device
from ..utils.exceptions import DISCOVERY_ALGORITHMS
from ..utils.memory import (
    check_memory_requirements,
    get_available_memory,
    get_memory_efficient_config,
    optimize_memory_usage,
    suggest_alternatives,
)
from .config import ConfigManager
from .debug import debug
from .utils import setup_logging, validate_model_name

console = Console()

# Discovery commands accept only discovery algorithms — not pruning/quantization
# methods (which also live in SUPPORTED_ALGORITHMS for the unified validator).
DISCOVERY_CHOICES = sorted(DISCOVERY_ALGORITHMS)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--config", "-c", help="Path to configuration file")
@click.pass_context
def cli(ctx, verbose, config):
    """CircuitKit: Circuit Discovery and Analysis for Transformer Models"""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["config"] = config

    # Setup logging
    setup_logging(verbose)

    if verbose:
        console.print("[bold blue]CircuitKit CLI[/bold blue] - Verbose mode enabled")


@cli.command()
@click.option(
    "--model",
    "-m",
    required=True,
    help="Model name or path (e.g., gpt2, meta-llama/Meta-Llama-3-8B)",
)
@click.option(
    "--algorithm",
    "-a",
    type=click.Choice(DISCOVERY_CHOICES),
    default="eap-ig",
    help="Discovery algorithm",
)
@click.option(
    "--task",
    "-t",
    default="ioi",
    help="Task name (ioi, sva, greater_than, capital_country, gender_bias, hypernymy, mmlu, glue_sst2)",
)
@click.option("--output", "-o", help="Output file path for results")
@click.option("--sparsity", "-s", type=float, default=0.3, help="Target sparsity (0.0-1.0)")
@click.option(
    "--level", "-l", type=click.Choice(["node", "neuron"]), default="node", help="Pruning level"
)
@click.option("--batch-size", "-b", type=int, default=4, help="Batch size")
@click.option("--ig-steps", type=int, default=5, help="Integrated gradients steps (for EAP-IG)")
@click.option(
    "--scope", type=click.Choice(["heads", "mlp", "both"]), default="both", help="Pruning scope"
)
@click.option(
    "--mlp-hook",
    type=click.Choice(["mlp_out", "post_act"]),
    default="mlp_out",
    help="MLP hook point for neuron-level discovery",
)
@click.option("--num-examples", type=int, default=128, help="Number of examples to generate")
@click.option(
    "--evaluate", is_flag=True, default=False, help="Evaluate circuit performance after discovery"
)
@click.option(
    "--random",
    is_flag=True,
    default=False,
    help="Evaluate random circuit performance for comparison",
)
@click.option(
    "--mlp1",
    is_flag=True,
    default=False,
    help="Score the intermediate d_mlp layer for neurons instead of d_model",
)
@click.option(
    "--chat-template-mode",
    type=click.Choice(["auto", "on", "off"]),
    default=None,
    help=(
        "Whether discovery prompts are wrapped in the model's chat template. "
        "'auto' wraps iff the model is instruction-tuned, 'on' always wraps, "
        "'off' never wraps. When unset, the task's own default applies."
    ),
)
def discover(
    model,
    algorithm,
    task,
    output,
    sparsity,
    level,
    batch_size,
    ig_steps,
    scope,
    mlp_hook,
    num_examples,
    evaluate,
    random,
    mlp1,
    chat_template_mode,
):
    """Run circuit discovery on a model using built-in task data generation"""
    from ..api import discover_circuit

    # Validate model name
    if not validate_model_name(model):
        console.print(f"[red]Error:[/red] Invalid model name: {model}")
        raise click.Abort()

    # Create configuration
    config = {
        "model": {"name": model, "precision": "bfloat16"},
        "pruning": {"target_sparsity": sparsity, "scope": scope, "random": random},
        "discovery": {
            "algorithm": algorithm,
            "task": task,
            "level": level,
            "batch_size": batch_size,
            "ig_steps": ig_steps,
            "evaluate": evaluate,
            "mlp_hook": mlp_hook,
            "data_params": {"num_examples": num_examples},
        },
    }

    if mlp1:
        config["discovery"]["mlp_hook"] = "post_act"

    # Thread the chat-template policy into the discovery config only when the
    # user explicitly set it; otherwise the task's own default applies.
    if chat_template_mode is not None:
        config["discovery"]["chat_template_mode"] = chat_template_mode

    # Set output path
    if not output:
        output = f"results/{algorithm}_{model.replace('/', '_')}_{task}_{level}.pt"
    config["output_path"] = output

    # Create output directory
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    console.print("[bold green]Starting circuit discovery...[/bold green]")
    console.print(f"Model: {model}")
    console.print(f"Algorithm: {algorithm.upper()}")
    console.print(f"Task: {task}")
    console.print(f"Level: {level}")
    console.print(f"Sparsity: {sparsity}")
    console.print(f"Output: {output}")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:
            task = progress.add_task("Running discovery...", total=100)

            # Run discovery
            result = discover_circuit(config)

            progress.update(task, completed=100)

        console.print("[bold green]✓ Discovery completed successfully![/bold green]")
        console.print(f"Results saved to: {output}")

        if isinstance(result, list):
            console.print(f"Pruned {len(result)} nodes")
        elif isinstance(result, dict):
            total_neurons = sum(len(neurons) for neurons in result.get("mlp", {}).values()) + sum(
                len(neurons) for neurons in result.get("attn", {}).values()
            )
            console.print(f"Pruned {total_neurons} neurons")

    except Exception as e:
        console.print(f"[red]Error during discovery:[/red] {str(e)}")
        raise click.Abort()


@cli.command()
@click.option("--model", "-m", required=True, help="Model name or path (e.g., gpt2)")
@click.option(
    "--task-yaml",
    "-t",
    required=True,
    type=click.Path(exists=True),
    help="Path to YAML task configuration file",
)
@click.option(
    "--algorithm",
    "-a",
    type=click.Choice(DISCOVERY_CHOICES),
    default="eap-ig",
    help="Discovery algorithm",
)
@click.option("--output", "-o", help="Output file path for results")
@click.option("--sparsity", "-s", type=float, default=0.3, help="Target sparsity (0.0-1.0)")
@click.option(
    "--level", "-l", type=click.Choice(["node", "neuron"]), default="node", help="Pruning level"
)
@click.option("--batch-size", "-b", type=int, default=4, help="Batch size")
@click.option("--num-examples", type=int, default=128, help="Number of examples")
@click.option("--evaluate", is_flag=True, default=False, help="Evaluate circuit after discovery")
@click.option("--random", is_flag=True, default=False, help="Evaluate random circuit")
def discover_yaml(
    model, task_yaml, algorithm, output, sparsity, level, batch_size, num_examples, evaluate, random
):
    """Run circuit discovery using a YAML task configuration"""
    from pathlib import Path

    from ..api import discover_circuit
    from ..tasks.yaml_loader import YAMLTaskLoader

    # Validate model name
    if not validate_model_name(model):
        console.print(f"[red]Error:[/red] Invalid model name: {model}")
        raise click.Abort()

    # Load task from YAML
    try:
        task_spec = YAMLTaskLoader.load(Path(task_yaml))
        console.print(f"[green]✓ Loaded task from YAML: {task_spec.name}[/green]")
    except Exception as e:
        console.print(f"[red]Error loading YAML task:[/red] {str(e)}")
        raise click.Abort()

    # Register task dynamically (idempotent: re-registering the same YAML is fine)
    from ..tasks.registry import is_task_registered, register_task

    if not is_task_registered(task_spec.name):
        register_task(task_spec)

    # Create configuration
    config = {
        "model": {"name": model, "precision": "bfloat16"},
        "pruning": {"target_sparsity": sparsity, "scope": "both", "random": random},
        "discovery": {
            "algorithm": algorithm,
            "task": task_spec.name,
            "level": level,
            "batch_size": batch_size,
            "evaluate": evaluate,
            "data_params": {"num_examples": num_examples},
        },
    }

    # Set output path
    if not output:
        yaml_stem = Path(task_yaml).stem
        output = f"results/{algorithm}_{model.replace('/', '_')}_{yaml_stem}_{level}.pt"
    config["output_path"] = output

    # Create output directory
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    console.print("[bold green]Starting YAML-based circuit discovery...[/bold green]")
    console.print(f"Model: {model}")
    console.print(f"Task: {task_spec.name}")
    console.print(f"YAML: {task_yaml}")
    console.print(f"Algorithm: {algorithm.upper()}")
    console.print(f"Output: {output}")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:
            task_progress = progress.add_task("Running discovery...", total=100)

            # Run discovery
            discover_circuit(config)

            progress.update(task_progress, completed=100)

        console.print("[bold green]✓ Discovery completed successfully![/bold green]")
        console.print(f"Results saved to: {output}")

    except Exception as e:
        console.print(f"[red]Error during discovery:[/red] {str(e)}")
        raise click.Abort()


@cli.command()
@click.option("--model", "-m", required=True, help="Model name or path")
@click.option("--artifact", "-a", required=True, help="Path to pruning artifact (.pt file)")
@click.option(
    "--task",
    "-t",
    default=None,
    help="Task name (e.g., ioi). Auto-derived from the artifact's _scores.json "
    "side-car if omitted; required only when that side-car is missing.",
)
@click.option(
    "--num-examples",
    "-n",
    type=int,
    default=None,
    help="Number of examples for faithfulness evaluation (default: from config/256)",
)
@click.option(
    "--precision",
    default="bfloat16",
    help="Torch dtype for the model (default: bfloat16)",
)
@click.option("--report-path", "-r", help="Path to write the JSON faithfulness result")
def evaluate(model, artifact, task, num_examples, precision, report_path):
    """Evaluate circuit faithfulness for a discovered/pruned artifact.

    Runs the 6-pillar faithfulness framework via ``evaluate_circuit`` and
    reports baseline / circuit / random performance. Discovery metadata
    (task, algorithm, level) is read from the ``_scores.json`` side-car that
    ``circuitkit discover`` writes next to the artifact.
    """
    import json

    from ..api import evaluate_circuit

    if not os.path.exists(artifact):
        console.print(f"[red]Error:[/red] Artifact file not found: {artifact}")
        raise click.Abort()

    if not report_path:
        report_path = f"results/evaluation_report_{model.replace('/', '_')}.json"

    # The artifact is self-describing: discover() writes a CircuitScores
    # side-car (<stem>_scores.json) carrying task / algorithm / level metadata.
    artifact_path = Path(artifact)
    scores_json = artifact_path.parent / (artifact_path.stem + "_scores.json")

    derived = {}
    if scores_json.exists():
        try:
            with open(scores_json, "r", encoding="utf-8") as f:
                meta = json.load(f)
            derived = {
                "task": meta.get("task"),
                "algorithm": meta.get("algorithm"),
                "level": meta.get("level"),
            }
            console.print(f"[dim]Read discovery metadata from {scores_json}[/dim]")
        except Exception as e:  # pragma: no cover - corrupt side-car
            console.print(f"[yellow]Warning:[/yellow] could not read {scores_json}: {e}")

    eval_task = task or derived.get("task")
    if not eval_task:
        console.print(
            "[red]Error:[/red] Could not determine the task. No '_scores.json' "
            f"side-car found next to {artifact}. Pass --task explicitly."
        )
        raise click.Abort()

    # Build a minimal valid config for evaluate_circuit. Only model.name and
    # discovery.{algorithm,task,level} are load-bearing here; everything else
    # falls back to DEFAULT_CONFIG inside load_and_validate_config().
    discovery_cfg = {"task": eval_task}
    if derived.get("algorithm"):
        discovery_cfg["algorithm"] = derived["algorithm"]
    if derived.get("level"):
        discovery_cfg["level"] = derived["level"]

    config = {
        "model": {"name": model, "precision": precision},
        "discovery": discovery_cfg,
    }
    if num_examples is not None:
        config["eval"] = {"num_examples": num_examples}

    console.print("[bold green]Starting circuit faithfulness evaluation...[/bold green]")
    console.print(f"Model: {model}")
    console.print(f"Artifact: {artifact}")
    console.print(f"Task: {eval_task}")
    console.print(f"Algorithm: {discovery_cfg.get('algorithm', '(default)')}")
    console.print(f"Level: {discovery_cfg.get('level', '(default)')}")
    console.print(f"Report: {report_path}")

    # Create output directory
    report_dir = os.path.dirname(report_path)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:
            prog_task = progress.add_task("Running evaluation...", total=100)

            result = evaluate_circuit(config, pruned_artifact_path=artifact)

            progress.update(prog_task, completed=100)

        result.to_json(report_path)

        console.print("[bold green]✓ Evaluation completed successfully![/bold green]")
        console.print(f"Patching score (P1): {result.patching_score}")
        console.print(f"Ablation score (P2): {result.ablation_score}")
        random_avg = result.metadata.get("random_avg")
        if random_avg is not None:
            console.print(f"Random-circuit avg:  {random_avg}")
        console.print(f"Report saved to: {report_path}")

    except Exception as e:
        console.print(f"[red]Error during evaluation:[/red] {str(e)}")
        raise click.Abort()


@cli.command()
@click.option("--limit", "-l", type=int, default=50, help="Limit number of models shown")
@click.option("--type", "-t", help="Filter by model type (e.g., GPT, Llama, OPT)")
@click.option("--size", "-s", help="Filter by model size (e.g., Small, Medium, Large)")
def list_models(limit, type, size):
    """List supported models from TransformerLens"""
    from .utils import get_model_info, get_supported_models

    console.print("[bold green]Supported Models (from TransformerLens):[/bold green]")

    try:
        # Get all available models
        all_models = get_supported_models()
        console.print(f"[dim]Found {len(all_models)} available models[/dim]")

        # Filter models if requested
        models_to_scan = all_models
        if limit and not type and not size:
            models_to_scan = all_models[:limit]
            console.print(f"[dim]Showing first {limit} models (use --limit to see more)[/dim]")

        filtered_models = []
        for model_name in models_to_scan:
            model_info = get_model_info(model_name)

            # Apply filters
            if type and type.lower() not in model_info["type"].lower():
                continue
            if size and size.lower() not in model_info["size"].lower():
                continue

            filtered_models.append(model_info)

        # Limit filtered results
        if (type or size) and limit and len(filtered_models) > limit:
            filtered_models = filtered_models[:limit]
            console.print(f"[dim]Showing first {limit} models (use --limit to see more)[/dim]")

        if not filtered_models:
            console.print("[yellow]No models found matching the criteria[/yellow]")
            return

        # Create table
        models_table = Table(show_header=True, header_style="bold magenta")
        models_table.add_column("Model Name", style="cyan", max_width=40)
        models_table.add_column("Type", style="green")
        models_table.add_column("Size", style="yellow")
        models_table.add_column("Layers", style="blue")
        models_table.add_column("d_model", style="magenta")

        # Add models to table
        for model_info in filtered_models:
            models_table.add_row(
                model_info["name"],
                model_info["type"],
                model_info["size"],
                str(model_info["layers"]),
                str(model_info["d_model"]),
            )

        console.print(models_table)

        if len(filtered_models) < len(all_models):
            console.print(
                f"[dim]Showing {len(filtered_models)} of {len(all_models)} total models[/dim]"
            )
            console.print("[dim]Use --type and --size filters to narrow results[/dim]")

    except Exception as e:
        console.print(f"[red]Error listing models: {e}[/red]")
        console.print("[yellow]Falling back to common models...[/yellow]")

        # Fallback to common models
        models_table = Table(show_header=True, header_style="bold magenta")
        models_table.add_column("Model Name", style="cyan")
        models_table.add_column("Type", style="green")
        models_table.add_column("Size", style="yellow")

        common_models = [
            ("gpt2", "GPT-2", "Small"),
            ("gpt2-medium", "GPT-2", "Medium"),
            ("gpt2-large", "GPT-2", "Large"),
            ("meta-llama/Meta-Llama-3-8B", "Llama", "8B"),
            ("meta-llama/Meta-Llama-3-70B", "Llama", "70B"),
        ]

        for name, model_type, model_size in common_models:
            models_table.add_row(name, model_type, model_size)

        console.print(models_table)


@cli.command()
@click.option("--model", "-m", required=True, help="Model name or path")
@click.option(
    "--tasks", "-t", required=True, help="Comma-separated task names (e.g., ioi,sva,greater_than)"
)
@click.option(
    "--algorithm",
    "-a",
    type=click.Choice(DISCOVERY_CHOICES),
    default="eap-ig",
    help="Discovery algorithm",
)
@click.option("--output", "-o", help="Output directory for results")
@click.option("--sparsity", "-s", type=float, default=0.3, help="Target sparsity")
@click.option(
    "--level", "-l", type=click.Choice(["node", "neuron"]), default="node", help="Pruning level"
)
@click.option("--num-examples", type=int, default=128, help="Number of examples per task")
@click.option(
    "--visualize", is_flag=True, default=True, help="Generate visualizations (default: True)"
)
@click.option(
    "--analyze", is_flag=True, default=True, help="Run statistical analysis (default: True)"
)
def transfer_matrix(
    model, tasks, algorithm, output, sparsity, level, num_examples, visualize, analyze
):
    """Build and analyze a cross-task transfer matrix.

    Discovers circuits on each source task and evaluates them on all target tasks,
    building an NxN matrix showing how well circuits transfer across tasks.

    Example:
        circuitkit transfer-matrix -m gpt2 -t ioi,sva,greater_than
    """
    from ..evaluation.transfer import TransferMatrix

    # Parse task list
    task_list = [t.strip() for t in tasks.split(",")]
    console.print(
        f"[bold green]Building transfer matrix for tasks:[/bold green] {', '.join(task_list)}"
    )

    # Set output directory
    if not output:
        output = f"results/transfer_matrix_{model.replace('/', '_')}"

    os.makedirs(output, exist_ok=True)

    console.print(f"Model: {model}")
    console.print(f"Tasks: {len(task_list)}")
    console.print(f"Expected discovery runs: {len(task_list)}")
    console.print(f"Expected evaluation runs: {len(task_list) ** 2}")
    console.print(f"Output directory: {output}")

    try:
        # Create transfer matrix builder
        tm = TransferMatrix(task_list)

        # Build discovery config template
        discovery_cfg = {
            "model": {"name": model, "precision": "bfloat16"},
            "pruning": {"target_sparsity": sparsity, "scope": "both", "random": False},
            "discovery": {
                "algorithm": algorithm,
                "level": level,
                "batch_size": 4,
                "data_params": {"num_examples": num_examples},
            },
        }

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:
            task_prog = progress.add_task("Building transfer matrix...", total=100)

            # Build matrix (this will call discover_circuit and evaluate_circuit)
            tm.build(
                model=None,  # Model is loaded internally by discover_circuit/evaluate_circuit
                discovery_cfg_template=discovery_cfg,
                device=get_device(),
                save_dir=output,
                skip_diagonal=False,
            )

            progress.update(task_prog, completed=100)

        console.print("\n[bold green]✓ Transfer matrix built successfully![/bold green]")

        # Print summary
        console.print("\n" + tm.summary(threshold=0.5))

        # Save JSON results
        json_path = Path(output) / "transfer_matrix_analysis.json"
        tm.to_json(json_path)
        console.print(f"[green]Analysis saved to {json_path}[/green]")

        # Generate visualizations
        if visualize:
            console.print("\n[bold blue]Generating visualizations...[/bold blue]")
            try:
                tm.visualize(output_dir=output)
                console.print(f"[green]✓ Visualizations saved to {output}[/green]")
            except ImportError as e:
                console.print(f"[yellow]⚠️  Visualization skipped: {e}[/yellow]")

        # Run statistical analysis
        if analyze:
            console.print("\n[bold blue]Running statistical analysis...[/bold blue]")
            try:
                stats = tm.statistical_analysis()
                console.print("[green]✓ Statistical analysis complete[/green]")

                # Print key statistics
                cors = stats.get("correlation_structure", {})
                console.print("\nCorrelation Structure:")
                console.print(f"  Diagonal Strength: {cors.get('diagonal_strength', 0):.4f}")
                console.print(f"  Symmetry: {cors.get('symmetry', 0):.4f}")
                console.print(f"  Sparsity: {cors.get('sparsity', 0):.4f}")

            except ImportError as e:
                console.print(f"[yellow]⚠️  Statistical analysis skipped: {e}[/yellow]")

        console.print(f"\n[bold green]All results saved to:[/bold green] {output}")

    except Exception as e:
        console.print(f"[red]Error building transfer matrix:[/red] {str(e)}")
        import traceback

        if console._options.get("verbose"):
            traceback.print_exc()
        raise click.Abort()


# Add debug commands
cli.add_command(debug)


@cli.command()
@click.option(
    "--model", "-m", help="Preferred model name (will auto-select if insufficient memory)"
)
@click.option(
    "--algorithm",
    "-a",
    type=click.Choice(DISCOVERY_CHOICES),
    default="eap-ig",
    help="Discovery algorithm",
)
@click.option(
    "--task",
    "-t",
    default="ioi",
    help="Task name (ioi, sva, greater_than, capital_country, gender_bias, hypernymy, mmlu, glue_sst2)",
)
@click.option("--output", "-o", help="Output file path for results")
@click.option("--sparsity", "-s", type=float, default=0.1, help="Target sparsity (0.0-1.0)")
@click.option(
    "--level", "-l", type=click.Choice(["node", "neuron"]), default="node", help="Pruning level"
)
@click.option(
    "--scope", type=click.Choice(["heads", "mlp", "both"]), default="heads", help="Pruning scope"
)
@click.option(
    "--check-memory", is_flag=True, help="Check memory requirements and suggest alternatives"
)
@click.option("--num-examples", type=int, default=128, help="Number of examples to generate")
@click.option(
    "--evaluate", is_flag=True, default=False, help="Evaluate circuit performance after discovery"
)
@click.option(
    "--random",
    is_flag=True,
    default=False,
    help="Evaluate random circuit performance for comparison",
)
@click.option(
    "--mlp1",
    is_flag=True,
    default=False,
    help="Score the intermediate d_mlp layer for neurons instead of d_model",
)
def discover_smart(
    model,
    algorithm,
    task,
    output,
    sparsity,
    level,
    scope,
    check_memory,
    num_examples,
    evaluate,
    random,
    mlp1,
):
    """Run memory-efficient circuit discovery with automatic model selection"""
    from ..api import discover_circuit

    if check_memory:
        # Just check memory and show suggestions
        console.print("[bold blue]Memory Status Check[/bold blue]")
        memory_info = get_available_memory()

        table = Table(title="Memory Status")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        for key, value in memory_info.items():
            table.add_row(key.replace("_", " ").title(), f"{value:.2f} GB")

        console.print(table)

        if model:
            if check_memory_requirements(model):
                console.print(f"[green]✅ {model} has sufficient memory[/green]")
            else:
                console.print(f"[red]❌ {model} requires more memory[/red]")
                alternatives = suggest_alternatives(model)
                console.print(
                    f"[yellow]💡 Suggested alternatives: {', '.join(alternatives)}[/yellow]"
                )
        return

    # Auto-select model if not specified
    if not model:
        model = "gpt2"  # Default to smallest model

    # Check memory and suggest alternatives if needed
    if not check_memory_requirements(model):
        console.print(f"[yellow]⚠️  Insufficient memory for {model}[/yellow]")
        alternatives = suggest_alternatives(model)
        console.print(f"[blue]🔄 Using alternative: {alternatives[0]}[/blue]")
        model = alternatives[0]

    # Get memory-efficient configuration
    config = get_memory_efficient_config(model, algorithm)

    # Set task and data params
    config["discovery"]["task"] = task
    config["discovery"]["data_params"] = {"num_examples": num_examples}

    # Override with user-specified options
    if output:
        config["output_path"] = output
    if sparsity:
        config["pruning"]["target_sparsity"] = sparsity
    if level:
        config["discovery"]["level"] = level
    if scope:
        config["pruning"]["scope"] = scope

    # Set evaluation options
    config["discovery"]["evaluate"] = evaluate
    config["pruning"]["random"] = random

    if mlp1:
        config["discovery"]["mlp_hook"] = "post_act"

    # Apply memory optimizations
    optimize_memory_usage()

    console.print(f"[bold green]🚀 Starting memory-efficient discovery with {model}[/bold green]")
    console.print(f"Algorithm: {algorithm}")
    console.print(f"Batch size: {config['discovery']['batch_size']}")
    console.print(f"Sparsity: {config['pruning']['target_sparsity']}")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Running discovery...", total=None)

            discover_circuit(config)

            progress.update(task, description="✅ Discovery complete!")

        console.print("[green]✅ Discovery completed successfully![/green]")
        console.print(f"Results saved to: {config['output_path']}")

    except Exception as e:
        console.print(f"[red]❌ Discovery failed: {e}[/red]")
        raise click.ClickException(f"Discovery failed: {e}")


@cli.command()
@click.option("--config", "-c", help="Path to configuration file")
def validate_config(config):
    """Validate a configuration file"""
    if not config:
        console.print("[red]Error:[/red] Configuration file path is required")
        raise click.Abort()

    if not os.path.exists(config):
        console.print(f"[red]Error:[/red] Configuration file not found: {config}")
        raise click.Abort()

    try:
        config_manager = ConfigManager(config)
        console.print("[bold green]✓ Configuration is valid![/bold green]")
        console.print(f"Loaded from: {config}")

        # Display configuration summary
        config_data = config_manager.config
        console.print("\n[bold]Configuration Summary:[/bold]")
        console.print(f"Model: {config_data['model']['name']}")
        console.print(f"Algorithm: {config_data['discovery']['algorithm']}")
        console.print(f"Level: {config_data['discovery']['level']}")
        console.print(f"Sparsity: {config_data['pruning']['target_sparsity']}")

    except Exception as e:
        console.print(f"[red]Error validating configuration:[/red] {str(e)}")
        raise click.Abort()


@cli.command()
@click.option("--model", "-m", required=True, help="Model name (e.g., gpt2)")
@click.option(
    "--pruned-model",
    "-p",
    required=True,
    type=click.Path(exists=True),
    help="Path to pruned model weights",
)
@click.option(
    "--circuit-scores",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="Path to circuit scores artifact (.pt file)",
)
@click.option("--task", "-t", default="ioi", help="Task name for training data")
@click.option("--lora-rank", type=int, default=8, help="LoRA rank (default: 8)")
@click.option("--epochs", type=int, default=3, help="Training epochs (default: 3)")
@click.option("--learning-rate", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
@click.option("--batch-size", "-b", type=int, default=4, help="Batch size")
@click.option(
    "--score-threshold", type=float, default=0.0, help="Only heal nodes with score >= threshold"
)
@click.option("--output", "-o", help="Output path for healed model")
@click.option("--device", default="cuda", help="Device to use (cuda/cpu)")
def heal(
    model,
    pruned_model,
    circuit_scores,
    task,
    lora_rank,
    epochs,
    learning_rate,
    batch_size,
    score_threshold,
    output,
    device,
):
    """
    Fine-tune a pruned model using circuit-guided LoRA soft healing.

    Example:
        circuitkit heal -m gpt2 -p pruned.pt -c scores.pt --task ioi
    """

    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformer_lens import HookedTransformer

    from circuitkit.applications.finetuning.soft_healing import CircuitLoRA

    from ..tasks.bootstrap import _bootstrap_builtin_tasks
    from ..tasks.registry import get_task, is_task_registered

    console.print("[bold green]Starting Circuit-Guided Soft Healing[/bold green]")
    console.print(f"Model: {model}")
    console.print(f"Pruned model: {pruned_model}")
    console.print(f"Circuit scores: {circuit_scores}")
    console.print(f"Task: {task}")
    console.print(f"LoRA rank: {lora_rank}")
    console.print(f"Epochs: {epochs}")
    console.print(f"Learning rate: {learning_rate}")

    try:
        # Load model
        console.print(f"\n[blue]Loading model: {model}[/blue]")
        model_obj = HookedTransformer.from_pretrained(model, device=device)

        # Load circuit scores
        console.print(f"[blue]Loading circuit scores from {circuit_scores}[/blue]")
        # weights_only=True: the scores file is untrusted user input (a plain
        # score dict); this blocks pickle-based RCE (CWE-502).
        artifact = torch.load(circuit_scores, map_location="cpu", weights_only=True)

        # Extract circuit scores
        if isinstance(artifact, dict):
            circuit_scores_dict = artifact.get("scores", artifact)
        else:
            circuit_scores_dict = artifact

        if not isinstance(circuit_scores_dict, dict):
            console.print(
                f"[red]Error:[/red] Circuit scores must be a dict, got {type(circuit_scores_dict)}"
            )
            raise click.Abort()

        console.print(f"Found {len(circuit_scores_dict)} circuit nodes above threshold")

        # Create CircuitLoRA
        console.print("\n[blue]Creating CircuitLoRA adapter...[/blue]")
        lora = CircuitLoRA(
            model=model_obj,
            circuit_scores=circuit_scores_dict,
            lora_rank=lora_rank,
            score_threshold=score_threshold,
        )

        lora_params = lora.get_lora_parameters()
        console.print(
            f"✓ LoRA adapter created with {sum(p.numel() for p in lora_params):,} parameters"
        )

        # Load training data
        console.print(f"\n[blue]Loading training data for task: {task}[/blue]")
        vocab_size = getattr(getattr(model, "config", None), "vocab_size", 50257)
        X_train = X_val = None
        try:
            _bootstrap_builtin_tasks()
            task_spec = get_task(task) if is_task_registered(task) else None
            if task_spec is not None:

                def _collect_tokens(split, n):
                    dl = task_spec.build_dataloader(
                        split=split, batch_size=batch_size, num_examples=n
                    )
                    batches = []
                    for batch in dl:
                        ids = (
                            batch[0]
                            if isinstance(batch, (list, tuple))
                            else batch.input_ids if hasattr(batch, "input_ids") else batch
                        )
                        if isinstance(ids, torch.Tensor):
                            batches.append(ids.long().cpu())
                        if sum(b.shape[0] for b in batches) >= n:
                            break
                    return torch.cat(batches, dim=0)[:n] if batches else None

                X_train = _collect_tokens("train", 100)
                try:
                    X_val = _collect_tokens("val", 20)
                except Exception:
                    X_val = X_train[:20] if X_train is not None else None
        except Exception as e:
            console.print(f"[yellow]⚠️  Could not load task data: {e}[/yellow]")

        if X_train is None:
            console.print("[yellow]⚠️  Using synthetic token data as fallback[/yellow]")
            X_train = torch.randint(0, vocab_size, (100, 64))
        if X_val is None:
            X_val = torch.randint(0, vocab_size, (20, X_train.shape[1]))

        train_dataset = TensorDataset(X_train)
        val_dataset = TensorDataset(X_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        console.print(f"✓ Loaded {len(X_train)} training and {len(X_val)} validation examples")

        # Train LoRA
        console.print(f"\n[blue]Training LoRA adapters ({epochs} epochs)...[/blue]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:
            task_progress = progress.add_task("Training...", total=100)

            metrics = lora.train(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs,
                learning_rate=learning_rate,
                weight_decay=0.01,
            )

            progress.update(task_progress, completed=100)

        console.print("\n[bold green]✓ Training complete![/bold green]")
        console.print(f"Best epoch: {metrics['best_epoch'] + 1}")
        if metrics["val_loss"]:
            console.print(f"Best validation loss: {metrics['best_val_loss']:.4f}")

        # Save healed model and LoRA state
        if not output:
            output = f"results/healed_{model.replace('/', '_')}.pt"

        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

        console.print(f"\n[blue]Saving healed model to {output}[/blue]")

        # Save LoRA state dict
        lora_state = lora.get_lora_state_dict()

        save_dict = {
            "model": model,
            "circuit_scores": circuit_scores_dict,
            "lora_state_dict": lora_state,
            "lora_config": {
                "rank": lora_rank,
                "alpha": lora.lora_alpha,
                "target_modules": lora.target_modules,
                "score_threshold": score_threshold,
            },
            "metrics": metrics,
            "training_config": {
                "epochs": epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "task": task,
            },
        }

        torch.save(save_dict, output)
        console.print(f"✓ Saved to {output}")

        console.print("\n[bold green]Soft healing complete![/bold green]")
        console.print("Report:")
        console.print(f"  LoRA parameters: {metrics['total_params']:,}")
        console.print(f"  Training epochs: {epochs}")
        console.print(f"  Best epoch: {metrics['best_epoch'] + 1}")

    except Exception as e:
        console.print(f"[red]Error during healing:[/red] {str(e)}")
        import traceback

        if click.get_current_context().obj.get("verbose"):
            traceback.print_exc()
        raise click.Abort()


@cli.command()
@click.option("--model", "-m", required=True, help="Model name or path")
@click.option("--circuit-scores", "-cs", required=True, help="Path to circuit scores JSON")
@click.option("--task", "-t", default="ioi", help="Task name")
@click.option("--source-examples", "-se", required=True, help="Path to source examples CSV")
@click.option("--target-examples", "-te", required=True, help="Path to target examples CSV")
@click.option("--coefficient", "-c", type=float, default=1.0, help="Steering strength (0.0-2.0)")
@click.option("--output", "-o", help="Output directory for results")
@click.option("--batch-size", "-b", type=int, default=32, help="Batch size for steering")
@click.option(
    "--metric",
    type=click.Choice(["logit_diff", "accuracy", "top_k"]),
    default="logit_diff",
    help="Metric to measure steering effect",
)
@click.option("--top-k", type=int, default=5, help="Top-k for metric calculation")
@click.option("--threshold", type=float, default=0.0, help="Minimum circuit score threshold")
@click.option("--analyze", is_flag=True, default=False, help="Run detailed analysis")
def steer(
    model,
    circuit_scores,
    task,
    source_examples,
    target_examples,
    coefficient,
    output,
    batch_size,
    metric,
    top_k,
    threshold,
    analyze,
):
    """
    Apply activation steering to modify model behavior.

    Learns steering vectors from source/target example pairs and applies them
    to steer model behavior during inference.

    Example:
        circuitkit steer -m gpt2 -cs circuits/gpt2_ioi_scores.json \\
                         -se data/ioi_source.csv -te data/ioi_target.csv \\
                         -c 1.0
    """
    try:
        import json

        import pandas as pd
        import torch
        from transformer_lens import HookedTransformer

        from circuitkit.applications.steering.steering import ActivationSteering

        from ..artifacts.scores import CircuitScores

        console.print("[bold blue]🎯 Activation Steering[/bold blue]")
        console.print(f"Model: {model}")
        console.print(f"Task: {task}")
        console.print(f"Steering Coefficient: {coefficient}")

        # Load circuit scores
        if not os.path.exists(circuit_scores):
            console.print(f"[red]Error:[/red] Circuit scores file not found: {circuit_scores}")
            raise click.Abort()

        scores = CircuitScores.from_json(Path(circuit_scores))
        console.print(f"[green]✓ Loaded circuit scores[/green] ({len(scores.node_scores)} nodes)")

        # Load examples
        if not os.path.exists(source_examples):
            console.print(f"[red]Error:[/red] Source examples not found: {source_examples}")
            raise click.Abort()
        if not os.path.exists(target_examples):
            console.print(f"[red]Error:[/red] Target examples not found: {target_examples}")
            raise click.Abort()

        src_df = pd.read_csv(source_examples)
        tgt_df = pd.read_csv(target_examples)

        source_texts = [{"text": text} for text in src_df["text"].tolist()]
        target_texts = [{"text": text} for text in tgt_df["text"].tolist()]

        console.print(
            f"[green]✓ Loaded examples[/green] ({len(source_texts)} source, {len(target_texts)} target)"
        )

        # Load model (auto-detects CUDA / MPS / CPU)
        device = get_device()
        console.print(f"Loading model ({device})...")
        model_obj = HookedTransformer.from_pretrained(model, device=device)

        # Initialize steering
        steering = ActivationSteering(model_obj, scores.node_scores, score_threshold=threshold)

        # Compute steering vectors
        console.print("\n[bold blue]Computing steering vectors...[/bold blue]")
        steering_vectors = steering.compute_steering_vector(
            source_texts, target_texts, batch_size=batch_size
        )

        # Get statistics
        stats = steering.get_steering_statistics()
        console.print("\n[bold]Steering Statistics:[/bold]")
        stats_table = Table(title="Node Steering Vectors")
        stats_table.add_column("Node", style="cyan")
        stats_table.add_column("Norm", style="green")
        stats_table.add_column("Shape", style="yellow")

        for node_name, node_stats in sorted(stats.items()):
            stats_table.add_row(
                node_name, f"{node_stats['steering_norm']:.4f}", str(node_stats["shape"])
            )

        console.print(stats_table)

        # Set output directory
        if not output:
            output = f"results/steering_{model.replace('/', '_')}_{task}"
        os.makedirs(output, exist_ok=True)

        # Run analysis if requested
        results = {"steering_vectors": {}, "statistics": stats}

        if analyze:
            console.print("\n[bold blue]Running detailed analysis...[/bold blue]")

            # Measure steering effect at different coefficients
            test_input = source_texts[0]["text"] if source_texts else "test"

            def metric_fn(logits):
                if metric == "logit_diff":
                    # Simple metric: difference between top 2 logits
                    top_logits, _ = torch.topk(logits[:, -1, :], 2)
                    return (top_logits[:, 0] - top_logits[:, 1]).mean().item()
                elif metric == "accuracy":
                    return logits.argmax(dim=-1).float().mean().item()
                else:  # top_k
                    return torch.topk(logits[:, -1, :], top_k)[0].mean().item()

            coefficients = [0.0, 0.25, 0.5, 0.75, 1.0, coefficient]
            steering_effects = steering.measure_steering_effect(
                test_input, metric_fn, coefficients=coefficients
            )

            console.print("\n[bold]Steering Effects at Different Coefficients:[/bold]")
            effects_table = Table(title="Coefficient vs Metric")
            effects_table.add_column("Coefficient", style="cyan")
            effects_table.add_column("Metric Value", style="green")

            for coeff, value in sorted(steering_effects.items()):
                effects_table.add_row(f"{coeff:.2f}", f"{value:.4f}")

            console.print(effects_table)

            # Analyze node importance
            node_importance = steering.analyze_steering_importance(
                test_input, metric_fn, steering_vectors
            )
            results["node_importance"] = node_importance

        # Save results
        results_file = Path(output) / "steering_results.json"

        # Convert tensor results to serializable format
        serializable_results = {
            "steering_vectors": {
                k: v.cpu().tolist() if isinstance(v, torch.Tensor) else v
                for k, v in results.get("steering_vectors", {}).items()
            },
            "statistics": {
                k: {
                    kk: float(vv) if isinstance(vv, (int, float, torch.Tensor)) else vv
                    for kk, vv in v.items()
                }
                for k, v in results["statistics"].items()
            },
            "node_importance": {
                k: float(v) if isinstance(v, (int, float, torch.Tensor)) else v
                for k, v in results.get("node_importance", {}).items()
            },
            "metadata": {
                "model": model,
                "task": task,
                "coefficient": coefficient,
                "num_nodes": len(steering_vectors),
                "num_source_examples": len(source_texts),
                "num_target_examples": len(target_texts),
            },
        }

        with open(results_file, "w") as f:
            json.dump(serializable_results, f, indent=2)

        console.print("\n[green]✅ Steering complete![/green]")
        console.print(f"Results saved to: {output}")

    except Exception as e:
        console.print(f"[red]Error during steering:[/red] {str(e)}")
        import traceback

        if click.get_current_context().obj.get("verbose"):
            traceback.print_exc()
        raise click.Abort()


@cli.command()
@click.option(
    "--models",
    "-m",
    multiple=True,
    default=["gpt2"],
    help="Model names to benchmark (e.g., gpt2, llama2)",
)
@click.option(
    "--tasks",
    "-t",
    multiple=True,
    default=["ioi"],
    help="Task names (ioi, sva, greater_than, capital_country)",
)
@click.option(
    "--algorithms",
    "-a",
    multiple=True,
    default=["eap", "eap-ig"],
    help=f'Discovery algorithms ({", ".join(DISCOVERY_CHOICES)})',
)
@click.option(
    "--interventions",
    "-i",
    multiple=True,
    default=["prune", "heal"],
    help="Interventions to benchmark (prune, heal, steer, quantize)",
)
@click.option(
    "--baselines",
    "-b",
    multiple=True,
    default=["magnitude", "wanda", "random"],
    help="Baselines to compare (magnitude, wanda, gptq, sparsegpt, random)",
)
@click.option(
    "--sparsity-levels",
    type=float,
    multiple=True,
    default=[0.1, 0.3, 0.5],
    help="Sparsity levels to test",
)
@click.option(
    "--output-dir", "-o", default="./benchmark_results", help="Output directory for results"
)
@click.option("--num-examples", type=int, default=100, help="Number of examples per task")
@click.option(
    "--report-format",
    type=click.Choice(["html", "markdown", "json", "latex"]),
    default="html",
    help="Report output format",
)
@click.pass_context
def benchmark(
    ctx,
    models,
    tasks,
    algorithms,
    interventions,
    baselines,
    sparsity_levels,
    output_dir,
    num_examples,
    report_format,
):
    """Run comprehensive benchmarks comparing circuit methods and baselines."""
    from ..benchmarks import CircuitBenchmark

    try:
        console.print("[bold blue]CircuitKit Benchmarking Suite[/bold blue]")
        console.print(f"Models: {', '.join(models)}")
        console.print(f"Tasks: {', '.join(tasks)}")
        console.print(f"Algorithms: {', '.join(algorithms)}")
        console.print(f"Interventions: {', '.join(interventions)}")
        console.print(f"Baselines: {', '.join(baselines)}")
        console.print(f"Output directory: {output_dir}\n")

        # Create benchmark instance
        bench = CircuitBenchmark(
            model_names=list(models) if models else ["gpt2"],
            tasks=list(tasks) if tasks else ["ioi"],
            device=get_device(),
            output_dir=output_dir,
            verbose=ctx.obj.get("verbose", False),
        )

        # Load models
        with console.status("[bold green]Loading models..."):
            bench.load_models()

        # Run discovery benchmark
        console.print("\n[bold]Running discovery benchmark...[/bold]")
        discovery_results = bench.run_discovery_benchmark(
            algorithms=list(algorithms) if algorithms else ["eap", "eap-ig"],
            num_examples=num_examples,
        )
        console.print(f"✓ Completed {discovery_results['num_runs']} discovery runs")

        # Run intervention benchmark
        console.print("\n[bold]Running intervention benchmark...[/bold]")
        intervention_results = bench.run_intervention_benchmark(
            interventions=list(interventions) if interventions else ["prune", "heal"],
        )
        console.print(f"✓ Completed {intervention_results['num_runs']} intervention runs")

        # Run baseline comparison
        console.print("\n[bold]Running baseline comparison...[/bold]")
        baseline_results = bench.compare_with_baselines(
            baselines=list(baselines) if baselines else ["magnitude", "wanda", "random"],
            sparsity_levels=list(sparsity_levels) if sparsity_levels else [0.1, 0.3, 0.5],
        )
        console.print(f"✓ Completed {baseline_results['num_runs']} baseline runs")

        # Save results
        console.print("\n[bold]Saving results...[/bold]")
        results_file = bench.save_results()
        console.print(f"✓ Results saved to {results_file}")

        # Generate report
        console.print(f"\n[bold]Generating {report_format} report...[/bold]")
        report_file = bench.generate_report(output_format=report_format)
        console.print(f"✓ Report saved to {report_file}")

        console.print("\n[bold green]✅ Benchmarking complete![/bold green]")
        console.print(f"Results directory: {output_dir}")

    except Exception as e:
        console.print(f"[red]Error during benchmarking:[/red] {str(e)}")
        if ctx.obj.get("verbose"):
            import traceback

            traceback.print_exc()
        raise click.Abort()


# ---------------------------------------------------------------------------
# `circuitkit data ...` subcommands  (worthiness validator + adapter dispatch)
# ---------------------------------------------------------------------------


@cli.group()
def data():
    """Custom-data tools: worthiness check, prepare contrastive pairs."""


@data.command("check")
@click.argument("source", type=str)
@click.option(
    "--shape",
    default=None,
    help="Force a specific dataset shape "
    "(qa/mcq/pairwise/conversational/instruction/forget_retain). "
    "Default: auto-detect.",
)
@click.option("--max-records", type=int, default=128, help="How many records to inspect.")
@click.option(
    "--model",
    default=None,
    help="Optional HF model name for baseline-signal + " "logit-difference checks.",
)
@click.option("--device", default="cpu", help="Device for the baseline-signal check (e.g. cuda:0).")
@click.option("--output", default=None, help="If set, write the structured report JSON here.")
@click.option(
    "--hf-subset",
    default=None,
    help="HF dataset config/subset name (e.g. high_school_world_history " "for cais/mmlu).",
)
@click.option(
    "--hf-split", default=None, help="HF split name (e.g. test, train). Default: heuristic."
)
def data_check(source, shape, max_records, model, device, output, hf_subset, hf_split):
    """Check whether SOURCE is worthy of circuit discovery.

    SOURCE can be:
      - a HF dataset name (e.g. 'cais/mmlu', 'tatsu-lab/alpaca')
      - a path to a CSV
      - a path to a NormalizedDataset JSON

    Prints a green/yellow/red verdict + per-check explanations + the
    list of algorithms safe to apply on the data.
    """
    raw = _resolve_source(source, hf_subset, hf_split, max_records)
    from ..data.auto_detect import auto_normalize, detect_shape
    from ..data.normalized import DatasetShape
    from ..data.worthiness import evaluate_worthiness

    forced = DatasetShape(shape) if shape else None
    detected = forced or detect_shape(raw)
    click.echo(f"detected shape: {detected.value}" + (" (forced)" if forced else ""))

    ds = auto_normalize(
        raw, max_records=max_records, name=source, source=source, force_shape=forced
    )
    click.echo(f"loaded {len(ds)} records from {source} (n_paired={ds.n_paired})")

    tokenizer = mod = None
    if model:
        click.echo(f"loading {model} on {device} for baseline-signal check ...")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model)
        mod = AutoModelForCausalLM.from_pretrained(model).to(device).eval()

    report = evaluate_worthiness(ds, tokenizer=tokenizer, model=mod)
    click.echo("")
    click.echo(report.render_terminal())

    if output:
        report.save_json(output)
        click.echo(f"\nSaved JSON report to {output}")


@data.command("prepare")
@click.argument("source", type=str)
@click.option("--shape", default=None, help="Force shape; default auto-detect.")
@click.option(
    "--strategy",
    default=None,
    help="Corruption strategy name (e.g. mcq_choice_swap, "
    "entity_swap, resample). Default: shape-specific.",
)
@click.option("--max-records", type=int, default=256)
@click.option("--output", required=True, help="Output JSON path.")
@click.option("--hf-subset", default=None)
@click.option("--hf-split", default=None)
def data_prepare(source, shape, strategy, max_records, output, hf_subset, hf_split):
    """Build a NormalizedDataset of (clean, corrupt) pairs from SOURCE.

    Auto-detects the dataset shape, runs the matching adapter, applies
    the default (or user-specified) corruption strategy, writes the
    resulting NormalizedDataset JSON.
    """
    raw = _resolve_source(source, hf_subset, hf_split, max_records)
    from ..data.auto_detect import auto_normalize, default_strategy_for, detect_shape
    from ..data.corruption import get_strategy
    from ..data.normalized import DatasetShape

    forced = DatasetShape(shape) if shape else None
    actual_shape = forced or detect_shape(raw)
    if actual_shape == DatasetShape.UNKNOWN:
        raise click.UsageError(
            f"Could not auto-detect shape for {source}. " "Pass --shape explicitly."
        )

    ds = auto_normalize(
        raw, max_records=max_records, name=source, source=source, force_shape=forced
    )
    chosen_strategy = strategy or default_strategy_for(actual_shape)
    if chosen_strategy is None and not ds.fully_paired:
        click.echo(
            f"shape '{actual_shape.value}' has no default strategy and "
            f"records are not natively paired. Pick one with "
            f"--strategy."
        )
        raise click.Abort()

    if chosen_strategy is not None and not ds.fully_paired:
        strat = get_strategy(chosen_strategy)()
        new_records = []
        for r in ds.records:
            if strat.name == "resample":
                new_records.append(strat.apply(r, pool=ds.records))
            else:
                new_records.append(strat.apply(r))
        ds.records = new_records

    ds.save_json(output)
    click.echo(
        f"Saved {len(ds)} records to {output} "
        f"(shape={actual_shape.value}, paired={ds.n_paired}, "
        f"strategy={chosen_strategy or 'native'})"
    )

@data.command("template")
@click.argument("source", type=click.Path(exists=True))
@click.option("--clean-prompt", required=True, help="Clean prompt template, e.g. 'The capital of {country} is'")
@click.option("--corrupt-prompt", default=None, help="Corrupt prompt template, e.g. 'The capital of {other_country} is'. Omit for ibcircuit/cdt.")
@click.option("--clean-answer", required=True, help="Clean answer template, e.g. '{capital}'")
@click.option("--corrupt-answer", default=None, help="Corrupt answer template, e.g. '{other_capital}'. Omit for ibcircuit/cdt.")
@click.option("--pairing-mode", type=click.Choice(["explicit", "auto_peer"]), default="explicit")
@click.option("--max-records", type=int, default=256)
@click.option("--output", required=True, help="Output NormalizedDataset JSON path.")
@click.option("--validate", "run_validation", is_flag=True, default=False, help="Run worthiness checks.")
@click.option(
    "--align-strategy",
    type=click.Choice(["filter", "pad_question", "none"]),
    default="filter",
    show_default=True,
    help=(
        "Token-alignment enforcement strategy. Ignored when --corrupt-prompt is omitted. "
        "'filter': drop misaligned pairs (safest). "
        "'pad_question': pad corrupt prompt up to clean length before pad_region_end. "
        "'none': no enforcement (use with metric=kl_divergence)."
    ),
)
@click.option(
    "--pad-region-end",
    default=None,
    help="Boundary string required when --align-strategy=pad_question (e.g. 'Answer:').",
)
@click.option(
    "--model",
    default=None,
    help=(
        "HF model name to load tokenizer from (required for --align-strategy=filter/pad_question). "
        "Omit to auto-downgrade align-strategy to 'none'."
    ),
)
@click.option(
    "--pair-padding-side",
    type=click.Choice(["left", "right"]),
    default="left",
    show_default=True,
    help="Padding side for contrastive pair batches. Ignored when --corrupt-prompt is omitted.",
)
def data_template(
    source,
    clean_prompt,
    corrupt_prompt,
    clean_answer,
    corrupt_answer,
    pairing_mode,
    max_records,
    output,
    run_validation,
    align_strategy,
    pad_region_end,
    model,
    pair_padding_side,
):
    """Build a dataset from SOURCE CSV using template substitution.

    Provide all four template options (--clean-prompt, --corrupt-prompt,
    --clean-answer, --corrupt-answer) to produce a fully-paired dataset
    for EAP/ACDC algorithms.

    Omit --corrupt-prompt and --corrupt-answer to produce a clean-only
    dataset for ibcircuit or cdt, without needing a separate CSV.
    """
    from ..data.template import clean_only_from_template, template_normalize

    is_paired = corrupt_prompt is not None and corrupt_answer is not None

    if not is_paired and (corrupt_prompt is not None or corrupt_answer is not None):
        raise click.UsageError(
            "Provide both --corrupt-prompt and --corrupt-answer for paired data, "
            "or omit both for clean-only output (ibcircuit/cdt)."
        )

    if is_paired:
        # ── Paired path: full template_normalize with alignment ───────────────
        tokenizer = None
        if align_strategy != "none":
            if model is None:
                original_strategy = align_strategy
                align_strategy = "none"
                warn_msg = (
                    f"Warning: --align-strategy='{original_strategy}' requires a tokenizer but "
                    f"--model was not provided. Downgrading to align-strategy='none'. "
                    f"Pass --model <hf_model_name> to enable alignment enforcement."
                )
                if pad_region_end:
                    warn_msg += f" --pad-region-end='{pad_region_end}' will be ignored."
                click.echo(warn_msg, err=True)
            else:
                try:
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(model)
                    click.echo(f"Loaded tokenizer from {model}")
                except Exception as e:
                    raise click.UsageError(
                        f"Could not load tokenizer from '{model}': {e}. "
                        f"Pass a valid HF model name or use --align-strategy=none."
                    )

        if align_strategy == "pad_question" and not pad_region_end:
            raise click.UsageError(
                "--pad-region-end is required when --align-strategy=pad_question. "
                "Provide the boundary string that separates the question from the answer "
                "(e.g. --pad-region-end 'Answer:')."
            )

        template_spec = {
            "clean_prompt": clean_prompt,
            "corrupt_prompt": corrupt_prompt,
            "clean_answer": clean_answer,
            "corrupt_answer": corrupt_answer,
        }
        ds = template_normalize(
            source,
            template_spec=template_spec,
            pairing_mode=pairing_mode,
            max_records=max_records,
            name=source,
            source=source,
            align_strategy=align_strategy,
            tokenizer=tokenizer,
            pad_region_end=pad_region_end,
        )
        if pair_padding_side != "left":
            ds.meta.setdefault("_alignment", {})["recommended_pair_padding_side"] = pair_padding_side

        _alignment = ds.meta.get("_alignment", {})
        if _alignment:
            kept = _alignment.get("kept", "?")
            total = _alignment.get("total_input", "?")
            dropped_nd = _alignment.get("dropped_nondiscriminative", 0)
            dropped_ma = _alignment.get("dropped_misaligned", 0)
            dropped_pf = _alignment.get("dropped_pad_failed", 0)
            click.echo(
                f"Alignment ({align_strategy}): {kept}/{total} kept  "
                f"[non-discriminative={dropped_nd}, misaligned={dropped_ma}, pad_failed={dropped_pf}]"
            )

        if run_validation:
            from ..data.worthiness import evaluate_worthiness
            report = evaluate_worthiness(ds)
            click.echo(report.render_terminal())

        ds.save_json(output)
        click.echo(
            f"Saved {len(ds)} template-paired records to {output} "
            f"(paired={ds.n_paired}, pair_padding_side={pair_padding_side})"
        )

    else:
        # ── Clean-only path: render clean side only ───────────────────────────
        template_spec = {
            "clean_prompt": clean_prompt,
            "clean_answer": clean_answer,
        }
        ds = clean_only_from_template(
            source,
            template_spec=template_spec,
            max_records=max_records,
            name=source,
            source=source,
        )

        if run_validation:
            from ..data.worthiness import evaluate_worthiness
            report = evaluate_worthiness(ds)
            click.echo(report.render_terminal())

        ds.save_json(output)
        click.echo(
            f"Saved {len(ds)} clean-only records to {output} "
            f"(compatible algorithms: ibcircuit, cdt)"
        )

    if run_validation:
        from ..data.worthiness import evaluate_worthiness
        report = evaluate_worthiness(ds)
        click.echo(report.render_terminal())

    ds.save_json(output)
    click.echo(f"Saved {len(ds)} template-paired records to {output} (paired={ds.n_paired})")

@data.command("clean-only")
@click.argument("source", type=click.Path(exists=True))
@click.option("--prompt-column", default="prompt", show_default=True,
              help="CSV column name for the clean prompt.")
@click.option("--answer-column", default="answer", show_default=True,
              help="CSV column name for the answer. Pass 'none' to skip (valid for CD-T).")
@click.option("--max-records", type=int, default=None,
              help="Truncate to this many records.")
@click.option("--output", required=True, help="Output NormalizedDataset JSON path.")
@click.option("--validate", "run_validation", is_flag=True, default=False,
              help="Run worthiness checks after loading.")
def data_clean_only(source, prompt_column, answer_column, max_records, output, run_validation):
    """Load SOURCE CSV as a clean-only dataset (no corrupt partner).

    Compatible with IBCircuit and CD-T discovery algorithms.
    """
    from ..data.clean_only import clean_only_normalize

    # Allow passing literal "none" on CLI to mean Python None
    _answer_col = None if answer_column.lower() == "none" else answer_column

    ds = clean_only_normalize(
        source,
        prompt_column=prompt_column,
        answer_column=_answer_col,
        max_records=max_records,
        name=source,
        source=source,
    )

    if run_validation:
        from ..data.worthiness import evaluate_worthiness
        report = evaluate_worthiness(ds)
        click.echo(report.render_terminal())

    ds.save_json(output)
    click.echo(
        f"Saved {len(ds)} clean-only records to {output} "
        f"(compatible algorithms: ibcircuit, cdt)"
    )

@data.command("shapes")
def data_shapes():
    """List supported dataset shapes + default strategies."""
    from ..data.auto_detect import list_supported_shapes

    for entry in list_supported_shapes():
        click.echo(
            f"  {entry['shape']:16}  {entry['adapter']:24}  "
            f"default_strategy={entry['default_strategy']}"
        )
        if entry["description"]:
            click.echo(f"      {entry['description'][:90]}")


@data.command("strategies")
def data_strategies():
    """List registered corruption strategies."""
    # Trigger self-registration via auto_detect's force-load path.
    from ..data import auto_detect  # noqa: F401
    from ..data.corruption import STRATEGY_REGISTRY

    for name, cls in sorted(STRATEGY_REGISTRY.items()):
        click.echo(f"  {name:22}  contract={cls.length_contract.value}")
        if cls.description:
            click.echo(f"      {cls.description[:90]}")


def _resolve_source(source, hf_subset, hf_split, max_records):
    """Source can be HF id, CSV path, or NormalizedDataset JSON."""
    from pathlib import Path

    p = Path(source)
    if p.exists():
        if p.suffix == ".csv":
            return str(p)
        if p.suffix == ".json":
            from ..data.normalized import NormalizedDataset

            return NormalizedDataset.load_json(str(p)).records
    # Otherwise assume HuggingFace dataset
    from datasets import load_dataset

    args = [source]
    if hf_subset:
        args.append(hf_subset)
    split = hf_split or "test"
    try:
        return list(load_dataset(*args, split=split, streaming=True).take(max_records or 64))
    except Exception:
        # Try common alternates
        for alt in ("train", "validation", "test"):
            try:
                return list(load_dataset(*args, split=alt, streaming=True).take(max_records or 64))
            except Exception:
                continue
        raise


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("artifact_path", type=click.Path(exists=True))
def inspect(artifact_path):
    """Inspect a circuit artifact (.pt) and print a summary."""
    from ..circuit import Circuit

    try:
        circuit = Circuit.from_artifact(artifact_path)
    except Exception as exc:
        console.print(f"[red]Error loading artifact:[/red] {exc}")
        raise click.Abort()

    table = Table(title=f"Circuit: {artifact_path}")
    table.add_column("Property", style="cyan")
    table.add_column("Value")
    table.add_row("Level", circuit.level)
    table.add_row("Nodes", str(len(circuit)))
    table.add_row("Algorithm", circuit.algorithm or "unknown")
    table.add_row("Task", circuit.task or "unknown")
    table.add_row("Model", circuit.model_name or "unknown")
    table.add_row("Scored Nodes", str(len(circuit.scores)))
    console.print(table)

    if circuit.scores:
        top = circuit.top_nodes(10)
        scores_table = Table(title="Top 10 Nodes by Score")
        scores_table.add_column("Node", style="cyan")
        scores_table.add_column("Score", justify="right")
        for name, score in top.items():
            scores_table.add_row(name, f"{score:.4f}")
        console.print(scores_table)


# ---------------------------------------------------------------------------
# prune
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--model", "-m", required=True, help="Model name or path (e.g., gpt2)")
@click.option(
    "--artifact", "-a", required=True, type=click.Path(exists=True),
    help="Path to circuit artifact (.pt)"
)
@click.option("--sparsity", "-s", type=float, default=0.3, show_default=True,
              help="Target sparsity (0.0–1.0)")
@click.option(
    "--scope", type=click.Choice(["heads", "mlp", "both"]), default="both", show_default=True,
    help="Which component type to prune"
)
@click.option("--output", "-o", required=True, help="Output checkpoint directory path")
@click.option(
    "--precision", default="bfloat16", show_default=True,
    help="Torch dtype for model loading"
)
def prune(model, artifact, sparsity, scope, output, precision):
    """Prune a model using a discovered circuit and export a HF checkpoint."""
    from .. import quick

    console.print(f"[bold green]Loading model:[/bold green] {model}")
    try:
        tl_model = quick.load_model(model, dtype=precision)
    except Exception as exc:
        console.print(f"[red]Error loading model:[/red] {exc}")
        raise click.Abort()

    from ..circuit import Circuit
    try:
        circuit = Circuit.from_artifact(artifact)
    except Exception as exc:
        console.print(f"[red]Error loading artifact:[/red] {exc}")
        raise click.Abort()

    console.print(
        f"Pruning [cyan]{model}[/cyan] at sparsity=[cyan]{sparsity}[/cyan] "
        f"scope=[cyan]{scope}[/cyan]"
    )
    try:
        pruned = quick.prune(tl_model, circuit, sparsity=sparsity, scope=scope)
        quick.export_checkpoint(pruned, circuit, output, intervention="pruning")
        console.print(f"[bold green]✓ Checkpoint exported to:[/bold green] {output}")
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise click.Abort()


# ---------------------------------------------------------------------------
# quantize
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--model", "-m", required=True, help="Model name or path")
@click.option(
    "--artifact", "-a", required=True, type=click.Path(exists=True),
    help="Path to circuit artifact (.pt)"
)
@click.option("--bits", type=int, default=4, show_default=True,
              help="Quantization bit-width")
@click.option("--high-fraction", type=float, default=0.3, show_default=True,
              help="Fraction of top layers kept at high precision")
@click.option(
    "--backend", type=click.Choice(["quanto", "llmcompressor"]), default="quanto",
    show_default=True, help="Quantization backend"
)
@click.option("--output", "-o", required=True, help="Output checkpoint directory path")
@click.option(
    "--precision", default="bfloat16", show_default=True,
    help="Torch dtype for model loading"
)
def quantize(model, artifact, bits, high_fraction, backend, output, precision):
    """Apply circuit-guided mixed-precision quantization and export a HF checkpoint."""
    from .. import quick

    console.print(f"[bold green]Loading model:[/bold green] {model}")
    try:
        tl_model = quick.load_model(model, dtype=precision)
    except Exception as exc:
        console.print(f"[red]Error loading model:[/red] {exc}")
        raise click.Abort()

    from ..circuit import Circuit
    try:
        circuit = Circuit.from_artifact(artifact)
    except Exception as exc:
        console.print(f"[red]Error loading artifact:[/red] {exc}")
        raise click.Abort()

    console.print(
        f"Quantizing [cyan]{model}[/cyan] to [cyan]{bits}[/cyan]-bit "
        f"(backend=[cyan]{backend}[/cyan], high_fraction=[cyan]{high_fraction}[/cyan])"
    )
    try:
        quantized = quick.quantize(
            tl_model, circuit,
            bits=bits,
            high_fraction=high_fraction,
            backend=backend,
        )
        quick.export_checkpoint(quantized, None, output, intervention="quantization")
        console.print(f"[bold green]✓ Checkpoint exported to:[/bold green] {output}")
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise click.Abort()


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--model", "-m", required=True, help="Model name or path")
@click.option(
    "--artifact", "-a", required=True, type=click.Path(exists=True),
    help="Path to circuit artifact (.pt)"
)
@click.option("--output", "-o", required=True, help="Output checkpoint directory path")
@click.option(
    "--intervention", type=click.Choice(["pruning", "quantization"]), default="pruning",
    show_default=True, help="Intervention type applied before export"
)
@click.option("--sparsity", "-s", type=float, default=0.3, show_default=True,
              help="Sparsity for pruning intervention")
@click.option(
    "--scope", type=click.Choice(["heads", "mlp", "both"]), default="both", show_default=True,
    help="Pruning scope (only used when intervention=pruning)"
)
@click.option(
    "--precision", default="bfloat16", show_default=True,
    help="Torch dtype for model loading"
)
def export(model, artifact, output, intervention, sparsity, scope, precision):
    """Apply an intervention to a model and export a HuggingFace checkpoint.

    For pruning: loads the circuit, prunes the model, exports.
    For quantization: loads the circuit, quantizes, exports.
    """
    from .. import quick
    from ..circuit import Circuit

    console.print(f"[bold green]Loading model:[/bold green] {model}")
    try:
        tl_model = quick.load_model(model, dtype=precision)
        circuit = Circuit.from_artifact(artifact)
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise click.Abort()

    console.print(
        f"Intervention: [cyan]{intervention}[/cyan] | "
        f"Model: [cyan]{model}[/cyan] | Output: [cyan]{output}[/cyan]"
    )
    try:
        if intervention == "pruning":
            intervened = quick.prune(tl_model, circuit, sparsity=sparsity, scope=scope)
        else:
            intervened = quick.quantize(tl_model, circuit)

        quick.export_checkpoint(
            intervened,
            circuit if intervention == "pruning" else None,
            output,
            intervention=intervention,
        )
        console.print(f"[bold green]✓ Exported to:[/bold green] {output}")
    except Exception as exc:
        console.print(f"[red]Error during export:[/red] {exc}")
        raise click.Abort()


# ---------------------------------------------------------------------------
# run (YAML pipeline)
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
def run(config_path):
    """Run a full pipeline from a YAML config file.

    The YAML config supports keys: model, task, precision, discovery,
    evaluate, applications, export, benchmark, visualize.
    See docs/guides/PIPELINE.md for the full format.
    """
    try:
        import yaml
    except ImportError:
        console.print("[red]Error:[/red] PyYAML is required. Install with: pip install pyyaml")
        raise click.Abort()

    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    except Exception as exc:
        console.print(f"[red]Error reading config:[/red] {exc}")
        raise click.Abort()

    from ..pipeline import Pipeline

    model_name = cfg.get("model")
    if not model_name:
        console.print("[red]Error:[/red] Config must contain a 'model' key.")
        raise click.Abort()

    task = cfg.get("task")
    precision = cfg.get("precision", "bfloat16")
    output_dir = cfg.get("output_dir", "./pipeline_output")

    # Handle custom data
    custom_data = cfg.get("custom_data")
    if custom_data:
        pipe = Pipeline.from_custom_data(
            model_name,
            custom_data["path"],
            clean_prompt=custom_data["clean_prompt"],
            clean_answer=custom_data["clean_answer"],
            corrupt_prompt=custom_data.get("corrupt_prompt"),
            corrupt_answer=custom_data.get("corrupt_answer"),
            precision=precision,
            output_dir=output_dir,
        )
    else:
        pipe = Pipeline(model_name, task=task, precision=precision, output_dir=output_dir)

    console.print(f"[bold green]Running pipeline:[/bold green] {config_path}")
    console.print(f"  Model: {model_name} | Task: {task or '(custom)'}")

    # --- Discovery ---
    disc = cfg.get("discovery", {})
    if disc:
        console.print("[cyan]Step: discovery[/cyan]")
        try:
            pipe.discover(
                algorithm=disc.get("algorithm", "eap-ig"),
                level=disc.get("level", "node"),
                sparsity=disc.get("sparsity", 0.3),
                n_examples=disc.get("n_examples", 128),
                batch_size=disc.get("batch_size", 4),
                scope=disc.get("scope", "both"),
            )
            console.print(f"  Circuit: {pipe._circuit!r}")
        except Exception as exc:
            console.print(f"[red]Discovery failed:[/red] {exc}")
            raise click.Abort()

    # --- Evaluate ---
    eval_cfg = cfg.get("evaluate", {})
    if eval_cfg and eval_cfg.get("enabled", True) and pipe._circuit is not None:
        console.print("[cyan]Step: evaluate[/cyan]")
        try:
            pipe.evaluate(
                pillars=eval_cfg.get("pillars"),
                n_examples=eval_cfg.get("n_examples", 256),
            )
        except Exception as exc:
            console.print(f"[yellow]Warning: evaluation failed:[/yellow] {exc}")

    # --- Applications ---
    for app in cfg.get("applications", []):
        app_type = app.get("type", "")
        console.print(f"[cyan]Step: {app_type}[/cyan]")
        try:
            if app_type == "prune":
                pipe.prune(
                    sparsity=app.get("sparsity", 0.3),
                    scope=app.get("scope", "both"),
                )
            elif app_type == "quantize":
                pipe.quantize(
                    bits=app.get("bits", 4),
                    high_fraction=app.get("high_fraction", 0.3),
                    backend=app.get("backend", "quanto"),
                )
            elif app_type == "selective_finetune":
                result = pipe.selective_finetune(
                    top_fraction=app.get("top_fraction", 0.2),
                    scope=app.get("scope", "both"),
                )
                console.print(f"  Selection result: {result}")
            else:
                console.print(f"[yellow]Unknown application type:[/yellow] {app_type!r}")
        except Exception as exc:
            console.print(f"[yellow]Warning: {app_type} failed:[/yellow] {exc}")

    # --- Export ---
    export_cfg = cfg.get("export", {})
    if export_cfg and export_cfg.get("path"):
        console.print("[cyan]Step: export[/cyan]")
        try:
            out = pipe.export(
                export_cfg["path"],
                intervention=export_cfg.get("intervention", "pruning"),
            )
            console.print(f"  Exported to: {out}")
        except Exception as exc:
            console.print(f"[yellow]Warning: export failed:[/yellow] {exc}")

    # --- Benchmark ---
    bench_cfg = cfg.get("benchmark", {})
    if bench_cfg and bench_cfg.get("enabled", False):
        console.print("[cyan]Step: benchmark[/cyan]")
        try:
            pipe.benchmark(
                tasks=bench_cfg.get("tasks"),
                limit=bench_cfg.get("limit"),
            )
        except Exception as exc:
            console.print(f"[yellow]Warning: benchmark failed:[/yellow] {exc}")

    # --- Visualize ---
    viz_cfg = cfg.get("visualize", {})
    if viz_cfg and viz_cfg.get("enabled", True) and pipe._circuit is not None:
        console.print("[cyan]Step: visualize[/cyan]")
        try:
            pipe.visualize(
                mode=viz_cfg.get("mode", "graph"),
                output=viz_cfg.get("output"),
            )
        except Exception as exc:
            console.print(f"[yellow]Warning: visualize failed:[/yellow] {exc}")

    console.print("[bold green]✓ Pipeline complete![/bold green]")
    pipe.summary()


def main():
    """Main entry point for CLI"""
    cli()


if __name__ == "__main__":
    main()

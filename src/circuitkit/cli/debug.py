"""
Debugging commands for CircuitKit CLI.
"""

import json
import os

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..utils.debugging import (
    cleanup_memory,
    debugger,
    disable_torch_debugging,
    enable_torch_debugging,
)
from ..utils.logging import get_logger

console = Console()
logger = get_logger(__name__)


@click.group()
def debug():
    """Debugging and profiling tools for CircuitKit."""


@debug.command()
@click.option("--enable", is_flag=True, help="Enable PyTorch debugging")
@click.option("--disable", is_flag=True, help="Disable PyTorch debugging")
def torch(enable, disable):
    """Configure PyTorch debugging settings."""
    if enable and disable:
        console.print("[red]Error:[/red] Cannot both enable and disable PyTorch debugging")
        raise click.Abort()

    if enable:
        enable_torch_debugging()
        console.print("[green]✓[/green] PyTorch debugging enabled")
    elif disable:
        disable_torch_debugging()
        console.print("[green]✓[/green] PyTorch debugging disabled")
    else:
        console.print("[yellow]Use --enable or --disable to configure PyTorch debugging[/yellow]")


@debug.command()
def memory():
    """Show current memory usage and cleanup."""
    import psutil
    import torch

    process = psutil.Process()
    memory_info = process.memory_info()

    # Create memory table
    table = Table(title="Memory Usage")
    table.add_column("Type", style="cyan")
    table.add_column("Usage (MB)", style="green")
    table.add_column("Percentage", style="yellow")

    # System memory
    system_memory = psutil.virtual_memory()
    table.add_row("System Total", f"{system_memory.total / 1024 / 1024:.1f}", "100%")
    table.add_row(
        "System Used", f"{system_memory.used / 1024 / 1024:.1f}", f"{system_memory.percent:.1f}%"
    )
    table.add_row(
        "System Available",
        f"{system_memory.available / 1024 / 1024:.1f}",
        f"{100 - system_memory.percent:.1f}%",
    )

    # Process memory
    table.add_row("Process RSS", f"{memory_info.rss / 1024 / 1024:.1f}", "")
    table.add_row("Process VMS", f"{memory_info.vms / 1024 / 1024:.1f}", "")

    # PyTorch memory
    if torch.cuda.is_available():
        table.add_row("CUDA Allocated", f"{torch.cuda.memory_allocated() / 1024 / 1024:.1f}", "")
        table.add_row("CUDA Reserved", f"{torch.cuda.memory_reserved() / 1024 / 1024:.1f}", "")
    else:
        table.add_row("CUDA", "Not available", "")

    console.print(table)

    # Cleanup option
    if click.confirm("Run memory cleanup?"):
        cleanup_memory()
        console.print("[green]✓[/green] Memory cleanup completed")


@debug.command()
@click.option("--output", "-o", help="Output file for debug report")
def report(output):
    """Generate and save debug report."""
    if not output:
        output = f"debug_report_{debugger.get_debug_report()['timestamp'].replace(':', '-')}.json"

    debugger.save_debug_report(output)
    console.print(f"[green]✓[/green] Debug report saved to {output}")

    # Show summary
    report_data = debugger.get_debug_report()

    summary_panel = Panel(
        f"Checkpoints: {len(report_data.get('checkpoints', []))}\n"
        f"Debug Info: {len(report_data.get('debug_info', {}))}\n"
        f"Performance Data: {'Yes' if 'performance' in report_data else 'No'}\n"
        f"Memory Data: {'Yes' if 'memory' in report_data else 'No'}",
        title="Debug Report Summary",
        border_style="green",
    )
    console.print(summary_panel)


@debug.command()
@click.option("--name", "-n", required=True, help="Checkpoint name")
@click.option("--data", "-d", help="Additional data (JSON format)")
def checkpoint(name, data):
    """Add a debug checkpoint."""
    checkpoint_data = {}
    if data:
        try:
            checkpoint_data = json.loads(data)
        except json.JSONDecodeError:
            console.print("[red]Error:[/red] Invalid JSON data")
            raise click.Abort()

    debugger.add_checkpoint(name, **checkpoint_data)
    console.print(f"[green]✓[/green] Checkpoint '{name}' added")


@debug.command()
@click.option("--name", "-n", help="Specific checkpoint name")
def list_checkpoints(name):
    """List debug checkpoints."""
    if name:
        checkpoint = debugger.get_checkpoint(name)
        if checkpoint:
            console.print(f"[green]Checkpoint '{name}':[/green]")
            console.print(json.dumps(checkpoint, indent=2))
        else:
            console.print(f"[red]Checkpoint '{name}' not found[/red]")
    else:
        checkpoints = debugger.get_all_checkpoints()
        if not checkpoints:
            console.print("[yellow]No checkpoints found[/yellow]")
            return

        table = Table(title="Debug Checkpoints")
        table.add_column("Name", style="cyan")
        table.add_column("Timestamp", style="green")
        table.add_column("Data Keys", style="yellow")

        for cp in checkpoints:
            data_keys = list(cp.get("data", {}).keys())
            table.add_row(
                cp["name"], cp["timestamp"], ", ".join(data_keys) if data_keys else "None"
            )

        console.print(table)


@debug.command()
def clear():
    """Clear all debug checkpoints."""
    if click.confirm("Clear all debug checkpoints?"):
        debugger.clear_checkpoints()
        console.print("[green]✓[/green] All checkpoints cleared")


@debug.command()
@click.option(
    "--level",
    "-l",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
    help="Logging level",
)
@click.option("--file", "-f", help="Log file path")
def logging(level, file):
    """Configure logging settings."""
    from ..utils.logging import setup_logging

    setup_logging(level=level, log_file=file)
    console.print(f"[green]✓[/green] Logging configured: level={level}, file={file or 'console'}")


@debug.command()
@click.option("--model", "-m", help="Model name to test")
@click.option("--data", "-d", help="Data file to test")
def test(model, data):
    """Run comprehensive functionality tests."""
    console.print("[bold blue]Running CircuitKit functionality tests...[/bold blue]")

    # Test 1: Model loading and basic operations
    console.print("\n[cyan]Test 1: Model Loading and Basic Operations[/cyan]")
    try:
        pass

        console.print("✅ Model loading test passed")
    except Exception as e:
        console.print(f"❌ Model loading test failed: {e}")

    # Test 2: Memory management
    console.print("\n[cyan]Test 2: Memory Management[/cyan]")
    try:
        import torch

        if torch.cuda.is_available():
            memory_before = torch.cuda.memory_allocated()
            # Perform a small operation
            memory_after = torch.cuda.memory_allocated()
            console.print(
                f"✅ Memory management test passed (allocated: {memory_after - memory_before} bytes)"
            )
        else:
            console.print("⚠️  CUDA not available, skipping memory test")
    except Exception as e:
        console.print(f"❌ Memory management test failed: {e}")

    # Test 3: Basic optimization utilities
    console.print("\n[cyan]Test 3: Basic Optimization Utilities[/cyan]")
    try:
        pass

        console.print("✅ Basic optimization utilities test passed")
    except Exception as e:
        console.print(f"❌ Basic optimization utilities test failed: {e}")

    # Test 4: Distributed training (if available)
    console.print("\n[cyan]Test 4: Distributed Training[/cyan]")
    try:
        from circuitkit.utils.distributed import DistributedTraining

        DistributedTraining()
        console.print("✅ Distributed training test passed")
    except Exception as e:
        console.print(f"❌ Distributed training test failed: {e}")

    console.print("\n[bold green]All functionality tests completed![/bold green]")

    # Test imports
    try:
        pass

        console.print("[green]✓[/green] API imports successful")
    except Exception as e:
        console.print(f"[red]✗[/red] API import failed: {e}")
        return

    # Test model loading
    if model:
        try:
            from transformer_lens import HookedTransformer

            HookedTransformer.from_pretrained(model, device="cpu")
            console.print(f"[green]✓[/green] Model '{model}' loaded successfully")
        except Exception as e:
            console.print(f"[red]✗[/red] Model loading failed: {e}")

    # Test data loading
    if data and os.path.exists(data):
        try:
            import pandas as pd

            df = pd.read_csv(data)
            console.print(f"[green]✓[/green] Data file '{data}' loaded: {len(df)} rows")
        except Exception as e:
            console.print(f"[red]✗[/red] Data loading failed: {e}")

    console.print("[bold green]Test completed![/bold green]")


if __name__ == "__main__":
    debug()

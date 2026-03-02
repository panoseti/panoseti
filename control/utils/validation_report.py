"""
utils/validation_report.py

Aggregates configuration validation errors, warnings, and informational messages.
Ensures silent operation during standard config loading, printing only when requested.
"""
from typing import List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

class ValidationReport:
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def add_info(self, msg: str) -> None:
        self.info.append(msg)

    def has_errors(self) -> bool:
        return len(self.errors) > 0

    def print_report(self, console: Console) -> None:
        """Prints a formatted validation report to the console."""
        if not self.errors and not self.warnings:
            console.print(Panel("[bold green]All Configurations Validated Successfully.[/bold green]", title="Validation Report"))
            return

        table = Table(title="Configuration Validation Issues", show_lines=True)
        table.add_column("Level", justify="center", style="bold")
        table.add_column("Message", justify="left")

        for err in self.errors:
            table.add_row("[red]ERROR[/red]", f"[red]{err}[/red]")
        for warn in self.warnings:
            table.add_row("[yellow]WARNING[/yellow]", f"[yellow]{warn}[/yellow]")
        for info in self.info:
            table.add_row("[blue]INFO[/blue]", f"[white]{info}[/white]")

        console.print(table)
import json
import os
import shutil
import sys
import typing
from pathlib import Path
from typing import Any, Type

import typer
import questionary
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from rich.console import Console
from rich.syntax import Syntax

from control.utils.paths import PanoPaths
from control.utils.pydantic_config_models import (
    DataConfig, ObsConfig, DaqConfig, NetworkConfig, DaemonConfig, FirmwareConfig, QuaboUids, ImageMode, PulseHeightMode, WpsConfig
)

console = Console()

app = typer.Typer(help="Interactive text-based configuration manager.")

CONFIG_TYPES = {
    "data_config.json": DataConfig,
    "obs_config.json": ObsConfig,
    "daq_config.json": DaqConfig,
    "network_config.json": NetworkConfig,
    "daemon_config.json": DaemonConfig,
    "firmware_config.json": FirmwareConfig,
    "quabo_uids.json": QuaboUids,
}

# A colorful style for questionary to make the UI pop
custom_style = questionary.Style([
    ('qmark', 'fg:#00ffff bold'),       # token in front of the question
    ('question', 'fg:#00ff00 bold'),    # question text
    ('answer', 'fg:#ff00ff bold'),      # submitted answer text behind the question
    ('pointer', 'fg:#ffff00 bold'),     # pointer used in select and checkbox prompts
    ('highlighted', 'fg:#ffff00 bold'), # pointed-at choice in select and checkbox
    ('selected', 'fg:#00ffff'),         # style for a selected item of a checkbox
    ('separator', 'fg:#888888'),        # separator in lists
    ('instruction', 'fg:#888888'),      # user instructions
    ('text', 'fg:#ffffff'),             # plain text
    ('disabled', 'fg:#858585 italic')   # disabled choices
])

def backup_file(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    
    # We want .bak-cfg-1 through .bak-cfg-4
    for i in range(3, 0, -1):
        src = path.with_suffix(path.suffix + f".bak-cfg-{i}")
        dst = path.with_suffix(path.suffix + f".bak-cfg-{i+1}")
        if src.exists() or src.is_symlink():
            if dst.exists() or dst.is_symlink():
                os.remove(dst)
            if src.is_symlink():
                os.rename(src, dst)
            else:
                shutil.move(str(src), str(dst))
                
    bak1 = path.with_suffix(path.suffix + ".bak-cfg-1")
    if path.is_symlink():
        target = os.readlink(path)
        os.symlink(target, bak1)
    else:
        shutil.copy(str(path), str(bak1))

def is_complex_type(annotation: Any) -> bool:
    return get_base_model(annotation) is not None

def get_base_model(annotation: Any) -> Type[BaseModel] | None:
    try:
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return annotation
    except TypeError:
        pass
    origin = typing.get_origin(annotation)
    if origin is not None:
        args = typing.get_args(annotation)
        for arg in args:
            try:
                if isinstance(arg, type) and issubclass(arg, BaseModel):
                    return arg
            except TypeError:
                pass
            inner = get_base_model(arg)
            if inner is not None:
                return inner
    return None

def edit_scalar(name: str, current_value: Any, field_info: FieldInfo, breadcrumb: str) -> Any:
    annotation = field_info.annotation
    origin = getattr(annotation, '__origin__', None)
    args = getattr(annotation, '__args__', ())
    prompt_text = f"[{breadcrumb}] Edit {name}:"

    if origin is getattr(typing, 'Literal', None) or type(annotation).__name__ == '_LiteralGenericAlias':
        choices = [str(a) for a in args]
        if origin is type(None) and "None" not in choices:
            choices.append("None")
        res = questionary.select(prompt_text, choices=choices, default=str(current_value) if current_value is not None else choices[0], style=custom_style).ask()
        if res is None or res == "None":
            return current_value
        return type(args[0])(res)

    if annotation is bool or (origin is type(None) and bool in args) or (origin is typing.Union and bool in args):
        res = questionary.confirm(f"[{breadcrumb}] Enable {name}?", default=bool(current_value), style=custom_style).ask()
        if res is None:
            return current_value
        return res

    def validate_input(val: str) -> bool | str:
        if not val and origin is type(None):
            return True
        try:
            if annotation is int or (origin is typing.Union and int in args):
                v = int(val)
                for meta in field_info.metadata:
                    if hasattr(meta, 'ge') and meta.ge is not None and v < meta.ge:
                        return f"Value must be >= {meta.ge}"
                    if hasattr(meta, 'le') and meta.le is not None and v > meta.le:
                        return f"Value must be <= {meta.le}"
            elif annotation is float or (origin is typing.Union and float in args):
                v = float(val)
                for meta in field_info.metadata:
                    if hasattr(meta, 'ge') and meta.ge is not None and v < meta.ge:
                        return f"Value must be >= {meta.ge}"
                    if hasattr(meta, 'le') and meta.le is not None and v > meta.le:
                        return f"Value must be <= {meta.le}"
        except ValueError:
            return "Invalid type."
        return True

    res = questionary.text(
        f"[{breadcrumb}] Edit {name} ({getattr(annotation, '__name__', str(annotation))}):",
        default=str(current_value) if current_value is not None else "",
        validate=validate_input,
        style=custom_style
    ).ask()

    if res is None:
        return current_value
    
    if res == "" and origin is type(None):
        return None

    try:
        def _check_type(ann, target_types):
            if ann in target_types: return True
            orig = getattr(ann, '__origin__', None)
            if orig in target_types: return True
            if orig is typing.Union: return any(_check_type(a, target_types) for a in getattr(ann, '__args__', ()))
            return False
            
        if _check_type(annotation, (list, dict)):
            import ast
            try:
                return ast.literal_eval(res)
            except Exception:
                import json
                try:
                    return json.loads(res)
                except Exception:
                    pass
                    
        if _check_type(annotation, (int,)):
            try: return int(res)
            except ValueError: pass
            
        if _check_type(annotation, (float,)):
            try: return float(res)
            except ValueError: pass
            
        return res
    except Exception:
        return current_value


def edit_model(model_class: Type[BaseModel], current_data: dict[str, Any], breadcrumb: str = "", indent: int = 0) -> tuple[str, dict[str, Any]]:
    import copy
    working_data = copy.deepcopy(current_data)
    prefix = "  " * indent
    last_selected = None
    while True:
        choices = []
        fields = getattr(model_class, 'model_fields', {})
        
        for field_name, field_info in fields.items():
            val = working_data.get(field_name)
            is_complex = is_complex_type(field_info.annotation)
            if is_complex:
                status = "[Disabled]" if val is None else "[Enabled]"
                choices.append(questionary.Choice(title=f"{prefix}{field_name} {status}", value=field_name))
            else:
                choices.append(questionary.Choice(title=f"{prefix}{field_name} (Current: {val})", value=field_name))
        
        is_extra_allowed = getattr(model_class, 'model_config', {}).get('extra') == 'allow'
        extra_keys = []
        if is_extra_allowed:
            for k, v in working_data.items():
                if k not in fields:
                    extra_keys.append(k)
                    if isinstance(v, dict) or isinstance(v, list):
                        choices.append(questionary.Choice(title=f"{prefix}{k} [Custom Block]", value=k))
                    else:
                        choices.append(questionary.Choice(title=f"{prefix}{k} (Current: {v}) [Custom Field]", value=k))
            
            if model_class == DataConfig:
                choices.append(questionary.Choice(title=f"{prefix}[n] Add custom image_*/pulse_height_* mode", value="__add_extra_mode__", shortcut_key="n"))
            elif model_class == ObsConfig:
                choices.append(questionary.Choice(title=f"{prefix}[n] Add custom WPS config", value="__add_wps__", shortcut_key="n"))
            else:
                choices.append(questionary.Choice(title=f"{prefix}[n] Add custom extra field", value="__add_generic_extra__", shortcut_key="n"))

        choices.append(questionary.Choice(title=f"{prefix}[v] Validate Current Draft", value="__validate__", shortcut_key="v"))
        choices.append(questionary.Choice(title=f"{prefix}[p] Preview Current JSON", value="__view__", shortcut_key="p"))
        choices.append(questionary.Choice(title=f"{prefix}[s] Save block and go up", value="__save__", shortcut_key="s"))
        choices.append(questionary.Choice(title=f"{prefix}[a] Save ALL edits and exit to menu", value="__save_all__", shortcut_key="a"))
        choices.append(questionary.Choice(title=f"{prefix}[d] Discard block edits and go up", value="__discard__", shortcut_key="d"))
        choices.append(questionary.Choice(title=f"{prefix}[x] Discard ALL edits and exit to menu", value="__discard_all__", shortcut_key="x"))
        choices.append(questionary.Choice(title=f"{prefix}[q] Quit immediately", value="__quit__", shortcut_key="q"))

        selected = questionary.select(
            f"[{breadcrumb}] Select field:", 
            choices=choices, 
            default=last_selected,
            use_shortcuts=True,
            style=custom_style
        ).ask()

        if selected is None or selected == "__quit__":
            confirm_quit = questionary.confirm("Are you sure you want to quit? Unsaved changes will be lost.", default=False, style=custom_style).ask()
            if confirm_quit:
                sys.exit(0)
            continue
            
        last_selected = selected
            
        if selected == "__save__":
            return ("__save__", working_data)
        if selected == "__save_all__":
            return ("__save_all__", working_data)
            
        if selected == "__discard__":
            return ("__discard__", current_data)
        if selected == "__discard_all__":
            return ("__discard_all__", current_data)
            
        if selected == "__validate__":
            try:
                model_class(**working_data)
                console.print(f"\n[green]Draft for {breadcrumb} is valid![/green]\n")
            except Exception as e:
                if hasattr(e, 'errors'):
                    console.print(f"\n[bold red]Validation Error(s):[/bold red]")
                    for err in e.errors():
                        loc = " -> ".join([str(part) for part in err["loc"]])
                        msg = err["msg"]
                        console.print(f"  [bold red]Field:[/bold red] {loc}")
                        console.print(f"  [bold red]Error:[/bold red] {msg}\n")
                else:
                    console.print(f"[red]Validation Error: {e}[/red]")
            input("Press Enter to return to the wizard...")
            continue
            
        if selected == "__view__":
            console.print(f"\n[bold magenta]--- Current State of {breadcrumb} ---[/bold magenta]")
            syntax = Syntax(json.dumps(working_data, indent=4), "json", theme="monokai", line_numbers=True)
            console.print(syntax)
            console.print("[bold magenta]-----------------------------------[/bold magenta]\n")
            input("Press Enter to return to the wizard...")
            continue
            
        if selected == "__add_extra_mode__":
            mode_type = questionary.select("Mode type:", choices=["image_", "pulse_height_"], style=custom_style).ask()
            if mode_type is None: sys.exit(0)
            if mode_type:
                mode_name = questionary.text("Enter mode name (e.g. 8bit, uhe):", style=custom_style).ask()
                if mode_name is None: sys.exit(0)
                if mode_name:
                    full_name = f"{mode_type}{mode_name}"
                    inner_model = ImageMode if mode_type == "image_" else PulseHeightMode
                    action, data = edit_model(inner_model, {}, breadcrumb=f"{breadcrumb} > {full_name}")
                    if action in ("__save__", "__save_all__"): working_data[full_name] = data
                    if action == "__save_all__": return action, working_data
                    if action == "__discard_all__": return action, current_data
            continue
            
        if selected == "__add_wps__":
            wps_name = questionary.text("Enter WPS name (e.g. wps2, wps-gh-runner):", style=custom_style).ask()
            if wps_name is None: sys.exit(0)
            if wps_name and wps_name.startswith("wps"):
                action, data = edit_model(WpsConfig, {}, breadcrumb=f"{breadcrumb} > {wps_name}", indent=indent + 1)
                if action in ("__save__", "__save_all__"): working_data[wps_name] = data
                if action == "__save_all__": return action, working_data
                if action == "__discard_all__": return action, current_data
            continue

        if selected == "__add_generic_extra__":
            key_name = questionary.text("Enter new field name:", style=custom_style).ask()
            if key_name is None: sys.exit(0)
            if key_name:
                val = questionary.text(f"Enter value for {key_name}:", style=custom_style).ask()
                if val is not None:
                    try:
                        working_data[key_name] = float(val) if "." in val else int(val)
                    except ValueError:
                        working_data[key_name] = val
            continue

        if selected in extra_keys:
            val = working_data[selected]
            if isinstance(val, dict):
                action = questionary.select(
                    f"[{breadcrumb}] Custom Block '{selected}'",
                    choices=[
                        questionary.Choice("Edit", value="Edit", shortcut_key="e"),
                        questionary.Choice("Disable (Delete) block", value="Disable (Delete) block", shortcut_key="d"),
                        questionary.Choice("Back", value="Back", shortcut_key="b")
                    ],
                    use_shortcuts=True,
                    style=custom_style
                ).ask()
                if action is None: sys.exit(0)
                if action == "Disable (Delete) block":
                    del working_data[selected]
                elif action == "Edit":
                    inner_model = None
                    if selected.startswith("image_"):
                        inner_model = ImageMode
                    elif selected.startswith("pulse_height_"):
                        inner_model = PulseHeightMode
                    elif selected.startswith("wps"):
                        inner_model = WpsConfig
                    
                    if inner_model:
                        action, data = edit_model(inner_model, working_data[selected], breadcrumb=f"{breadcrumb} > {selected}", indent=indent + 1)
                        if action in ("__save__", "__save_all__") or action == "__discard__": working_data[selected] = data
                        if action == "__save_all__": return action, working_data
                        if action == "__discard_all__": return action, current_data
                    else:
                        console.print(f"[yellow]Cannot edit unknown arbitrary dict {selected}[/yellow]")
            else:
                new_val = questionary.text(f"[{breadcrumb}] Edit {selected}:", default=str(val), style=custom_style).ask()
                if new_val is None: sys.exit(0)
                if new_val is not None:
                    try:
                        working_data[selected] = float(new_val) if "." in new_val else int(new_val)
                    except ValueError:
                        working_data[selected] = new_val
            continue

        field_info = fields[selected]
        is_complex = is_complex_type(field_info.annotation)

        if not is_complex:
            working_data[selected] = edit_scalar(selected, working_data.get(selected), field_info, breadcrumb=f"{breadcrumb} > {selected}")
            try:
                model_class(**working_data)
            except Exception as e:
                if hasattr(e, 'errors'):
                    for err in e.errors():
                        if err["loc"] and str(err["loc"][0]) == selected:
                            console.print(f"\n[bold red]Validation Error on '{selected}':[/bold red] {err['msg']}")
                            input("Press Enter to continue...")
                            break
        else:
            val = working_data.get(selected)
            if val is None:
                action = questionary.select(
                    f"[{breadcrumb}] Block '{selected}' is Disabled",
                    choices=[
                        questionary.Choice("Enable (Create) block", value="Enable (Create) block", shortcut_key="e"),
                        questionary.Choice("Back", value="Back", shortcut_key="b")
                    ],
                    use_shortcuts=True,
                    style=custom_style
                ).ask()
                if action is None: sys.exit(0)
                if action == "Enable (Create) block":
                    ann = field_info.annotation
                    origin = getattr(ann, '__origin__', None)
                    args = getattr(ann, '__args__', ())
                    if origin is list:
                        working_data[selected] = []
                    else:
                        inner_model = ann if hasattr(ann, 'model_fields') else args[0]
                        action, data = edit_model(inner_model, {}, breadcrumb=f"{breadcrumb} > {selected}", indent=indent + 1)
                        if action in ("__save__", "__save_all__") or action == "__discard__": working_data[selected] = data
                        if action == "__save_all__": return action, working_data
                        if action == "__discard_all__": return action, current_data
            else:
                ann = field_info.annotation
                origin = getattr(ann, '__origin__', None)
                
                # Check for list of BaseModel
                is_list = origin is list
                
                if is_list:
                    action_choices = [
                        questionary.Choice("Edit list items", value="Edit list items", shortcut_key="e"),
                        questionary.Choice("Disable (Delete) block", value="Disable (Delete) block", shortcut_key="d"),
                        questionary.Choice("Back", value="Back", shortcut_key="b")
                    ]
                else:
                    action_choices = [
                        questionary.Choice("Edit block", value="Edit block", shortcut_key="e"),
                        questionary.Choice("Disable (Delete) block", value="Disable (Delete) block", shortcut_key="d"),
                        questionary.Choice("Back", value="Back", shortcut_key="b")
                    ]
                    
                action = questionary.select(
                    f"[{breadcrumb}] Block '{selected}'",
                    choices=action_choices,
                    use_shortcuts=True,
                    style=custom_style
                ).ask()
                if action is None: sys.exit(0)
                
                if action == "Edit block" or action == "Edit list items":
                    ann = field_info.annotation
                    origin = getattr(ann, '__origin__', None)
                    args = getattr(ann, '__args__', ())
                    if origin is list:
                        inner_model = args[0]
                        last_list_sel = None
                        while True:
                            list_choices = [questionary.Choice(f"{prefix}  Item {i}", value=f"Item {i}") for i in range(len(working_data[selected]))]
                            list_choices.extend([
                                questionary.Choice(title=f"{prefix}  [n] Add new item", value="[+] Add new item", shortcut_key="n"),
                                questionary.Choice(title=f"{prefix}  [r] Remove item", value="[-] Remove item", shortcut_key="r"),
                                questionary.Choice(title=f"{prefix}  [b] Back", value="Back", shortcut_key="b")
                            ])
                            list_sel = questionary.select(
                                f"[{breadcrumb} > {selected}] Select list item:", 
                                choices=list_choices, 
                                default=last_list_sel,
                                use_shortcuts=True,
                                style=custom_style
                            ).ask()
                            if list_sel is None: sys.exit(0)
                            last_list_sel = list_sel
                            
                            if "Back" in list_sel:
                                break
                            elif "[+] Add new item" in list_sel:
                                action, data = edit_model(inner_model, {}, breadcrumb=f"{breadcrumb} > {selected} [NEW]")
                                if action in ("__save__", "__save_all__"):
                                    working_data[selected].append(data)
                                if action == "__save_all__": return action, working_data
                                if action == "__discard_all__": return action, current_data
                            elif "[-] Remove item" in list_sel:
                                if len(working_data[selected]) > 0:
                                    rm_idx = questionary.select("Select item to remove:", choices=[str(i) for i in range(len(working_data[selected]))], style=custom_style).ask()
                                    if rm_idx is None: sys.exit(0)
                                    if rm_idx:
                                        working_data[selected].pop(int(rm_idx))
                            else:
                                idx = int(list_sel.split("Item ")[1])
                                action, data = edit_model(inner_model, working_data[selected][idx], breadcrumb=f"{breadcrumb} > {selected} [{idx}]", indent=indent + 2)
                                if action in ("__save__", "__save_all__") or action == "__discard__": working_data[selected][idx] = data
                                if action == "__save_all__": return action, working_data
                                if action == "__discard_all__": return action, current_data
                    else:
                        inner_model = ann if hasattr(ann, 'model_fields') else args[0]
                        action, data = edit_model(inner_model, working_data[selected], breadcrumb=f"{breadcrumb} > {selected}", indent=indent + 1)
                        if action in ("__save__", "__save_all__") or action == "__discard__": working_data[selected] = data
                        if action == "__save_all__": return action, working_data
                        if action == "__discard_all__": return action, current_data
                elif action == "Disable (Delete) block":
                    working_data[selected] = None

    return "__save__", working_data

def discover_templates(config_dir: Path, target_filename: str) -> list[Path]:
    """Scans immediate subdirectories of config_dir for matching filenames."""
    templates = []
    if not config_dir.exists():
        return templates
    for item in config_dir.iterdir():
        if item.is_dir():
            for sub_item in item.glob(f"{target_filename}*"):
                if sub_item.is_file() or sub_item.is_symlink():
                    templates.append(sub_item)
    return templates

@app.command("edit")
def edit(tui: bool = typer.Option(False, "--tui", "-t", help="Use experimental Textual TUI interface")) -> None:
    """Interactive configuration manager."""
    if tui:
        try:
            from control.tui_app import ConfigManagerApp
            tui_app = ConfigManagerApp()
            tui_app.run()
            return
        except ImportError as e:
            console.print(f"[red]Failed to load TUI: {e}[/red]")
            console.print("[yellow]Falling back to basic questionary interface.[/yellow]")
        
    try:
        while True:
            # We loop the main wizard so users can edit multiple files or return from views
            if not _run_edit():
                break
    except KeyboardInterrupt:
        sys.exit(0)

def _run_edit() -> bool:
    """Returns True if the wizard should loop again, False to exit."""
    config_dir = PanoPaths.config_dir()
    
    console.print(f"\n[bold magenta]=== PSETI Configuration Manager ===[/bold magenta]")
    console.print(f"Workspace ([bold]PSETI_CONFIG[/bold]): [cyan]{config_dir}[/cyan]")
    if "PSETI_CONFIG" in os.environ:
        console.print("[dim](Loaded from environment variables / .env)[/dim]")
    print("")

    config_choices = []
    for ct in CONFIG_TYPES.keys():
        file_path = config_dir / ct
        abs_str = str(file_path.absolute())
        if not file_path.exists() and not file_path.is_symlink():
            config_choices.append(questionary.Choice(title=[("class:disabled", f"{ct}  ({abs_str})  [Missing]")], value=ct))
        else:
            config_choices.append(questionary.Choice(title=f"{ct}  ({abs_str})", value=ct))
            
    config_choices.append(questionary.Choice(title="[q] Cancel / Quit", value="Cancel / Quit", shortcut_key="q"))
    selected_type = questionary.select("Select config file to manage:", choices=config_choices, use_shortcuts=True, style=custom_style).ask()
    
    if selected_type is None or selected_type == "Cancel / Quit":
        return False
        
    model_class = CONFIG_TYPES[selected_type]
    file_path = config_dir / selected_type
    
    target_path_for_edit = file_path

    while True:
        action = None
        
        if file_path.is_symlink():
            target = os.readlink(file_path)
            abs_target = file_path.parent / target
            if abs_target.exists():
                console.print(f"State: [yellow]Valid Symlink[/yellow] (-> {target})")
                action = questionary.select(
                    "What would you like to do?",
                    choices=[
                        questionary.Choice("[e] Edit Target", value="Edit Target", shortcut_key="e"),
                        questionary.Choice("[v] View File contents", value="View File contents", shortcut_key="v"),
                        questionary.Choice("[c] Change Symlink (Choose Template)", value="Change Symlink (Choose Template)", shortcut_key="c"),
                        questionary.Choice("[b] Break Symlink (Copy to standalone)", value="Break Symlink (Copy to standalone)", shortcut_key="b"),
                        questionary.Choice("[q] Back", value="Back", shortcut_key="q")
                    ],
                    use_shortcuts=True,
                    style=custom_style
                ).ask()
                if action == "Edit Target":
                    target_path_for_edit = abs_target
                    break
                elif action == "View File contents":
                    _view_file(abs_target)
                    continue
            else:
                console.print(f"State: [red]Broken Symlink[/red] (-> {target})")
                action = questionary.select(
                    "What would you like to do?",
                    choices=[
                        questionary.Choice("[f] Fix Symlink (Choose Template)", value="Fix Symlink (Choose Template)", shortcut_key="f"),
                        questionary.Choice("[d] Delete and create standalone", value="Delete and create standalone", shortcut_key="d"),
                        questionary.Choice("[q] Back", value="Back", shortcut_key="q")
                    ],
                    use_shortcuts=True,
                    style=custom_style
                ).ask()
        elif file_path.exists():
            console.print(f"State: [green]Standalone File[/green]")
            action = questionary.select(
                "What would you like to do?",
                choices=[
                    questionary.Choice("[e] Edit", value="Edit", shortcut_key="e"),
                    questionary.Choice("[v] View File contents", value="View File contents", shortcut_key="v"),
                    questionary.Choice("[r] Replace with Symlink (Choose Template)", value="Replace with Symlink (Choose Template)", shortcut_key="r"),
                    questionary.Choice("[q] Back", value="Back", shortcut_key="q")
                ],
                use_shortcuts=True,
                style=custom_style
            ).ask()
            if action == "Edit":
                break
            if action == "View File contents":
                _view_file(file_path)
                continue
        else:
            console.print(f"State: [dim]Missing[/dim]")
            action = questionary.select(
                "What would you like to do?",
                choices=[
                    questionary.Choice("[s] Create from Template (Symlink)", value="Create from Template (Symlink)", shortcut_key="s"),
                    questionary.Choice("[c] Create from Template (Copy)", value="Create from Template (Copy)", shortcut_key="c"),
                    questionary.Choice("[n] Create from scratch", value="Create from scratch", shortcut_key="n"),
                    questionary.Choice("[q] Back", value="Back", shortcut_key="q")
                ],
                use_shortcuts=True,
                style=custom_style
            ).ask()
            if action == "Create from scratch":
                break

        if action is None or action == "Back":
            return True
            
        if action and ("Choose Template" in action or "Template" in action):
            templates = discover_templates(config_dir, selected_type)
            tpl_choices = [questionary.Choice(str(t.relative_to(config_dir)), value=str(t.relative_to(config_dir))) for t in templates]
            tpl_choices.append(questionary.Choice("[c] [Enter custom path...]", value="[Enter custom path...]", shortcut_key="c"))
            tpl_choices.append(questionary.Choice("[q] Back", value="Back", shortcut_key="q"))
            
            tpl_sel = questionary.select("Select template file:", choices=tpl_choices, use_shortcuts=True, style=custom_style).ask()
            if tpl_sel is None or tpl_sel == "Back":
                continue
                
            if tpl_sel == "[Enter custom path...]":
                custom_path_str = questionary.path("Enter path to template file:", style=custom_style).ask()
                if not custom_path_str: continue
                tpl_path = Path(custom_path_str).expanduser()
                if not tpl_path.exists() or not tpl_path.is_file():
                    console.print(f"[red]File {custom_path_str} does not exist or is not a file.[/red]")
                    input("Press Enter to continue...")
                    continue
            else:
                tpl_path = config_dir / tpl_sel
            
            if "Symlink" in action:
                if file_path.exists() or file_path.is_symlink():
                    backup_file(file_path)
                    os.remove(file_path)
                os.symlink(tpl_path, file_path)
                console.print(f"[green]Symlink created: {file_path.name} -> {tpl_path}[/green]")
                target_path_for_edit = tpl_path
                
                do_edit = questionary.confirm("Do you want to edit it now?", style=custom_style).ask()
                if not do_edit:
                    continue
                break
                    
            elif "Copy" in action:
                if file_path.exists() or file_path.is_symlink():
                    backup_file(file_path)
                    os.remove(file_path)
                shutil.copy(tpl_path, file_path)
                console.print(f"[green]Copied template to standalone file: {file_path}[/green]")
                target_path_for_edit = file_path
                break

        elif action == "Delete and create standalone" or action == "Break Symlink (Copy to standalone)":
            if file_path.is_symlink():
                backup_file(file_path)
                target = os.readlink(file_path)
                abs_target = file_path.parent / target
                os.remove(file_path)
                if "Copy to standalone" in action and abs_target.exists():
                    shutil.copy(abs_target, file_path)
                    console.print(f"[green]Symlink broken. Copied target to standalone file: {file_path}[/green]")
                else:
                    console.print(f"[green]Symlink deleted. Starting fresh.[/green]")
            target_path_for_edit = file_path
            break

    current_data = {}
    if target_path_for_edit.exists():
        try:
            with open(target_path_for_edit, 'r') as f:
                raw_json = json.load(f)
            
            # Best-effort load: apply defaults and aliases without crashing on validation errors
            model_instance = model_class.model_construct(**raw_json)
            current_data = model_instance.model_dump(exclude_none=True, by_alias=True)
            console.print(f"[dim]Loaded: {target_path_for_edit}[/dim]")
        except Exception as e:
            console.print(f"[red]Error loading file (Malformed JSON): {e}[/red]")
            start_fresh = questionary.confirm("Do you want to start fresh? (This will overwrite the corrupt file upon saving)", style=custom_style).ask()
            if not start_fresh:
                return True
            current_data = {}

    # Drop into the editor with breadcrumbs
    action, current_data = edit_model(model_class, current_data, breadcrumb=selected_type)
    
    if action in ("__discard__", "__discard_all__"):
        console.print("[dim]Edits discarded. File was not saved.[/dim]")
        return True
    
    validated_data = current_data
    try:
        model_instance = model_class(**current_data)
        validated_data = model_instance.model_dump(exclude_none=True, by_alias=True)
        console.print("[green]Validation successful![/green]")
    except Exception as e:
        if hasattr(e, 'errors'):
            console.print(f"\n[bold red]Validation Error(s):[/bold red]")
            for err in e.errors():
                loc = " -> ".join([str(part) for part in err["loc"]])
                msg = err["msg"]
                console.print(f"  [bold red]Field:[/bold red] {loc}")
                console.print(f"  [bold red]Error:[/bold red] {msg}\n")
        else:
            console.print(f"[red]Validation Error: {e}[/red]")
        save_anyway = questionary.confirm("Save anyway? (Will result in invalid config)", style=custom_style).ask()
        if not save_anyway:
            return True
            
        # Best-effort serialization for invalid configs (applies defaults and aliases without validating)
        console.print("[yellow]Performing best-effort serialization of invalid config...[/yellow]")
        model_instance = model_class.model_construct(**current_data)
        validated_data = model_instance.model_dump(exclude_none=True, by_alias=True)

    target_path_for_edit.parent.mkdir(parents=True, exist_ok=True)
    if target_path_for_edit.exists() or target_path_for_edit.is_symlink():
        backup_file(target_path_for_edit)
    
    with open(target_path_for_edit, 'w') as f:
        json.dump(validated_data, f, indent=4)
    console.print(f"[bold green]Saved successfully to {target_path_for_edit}[/bold green]")
    input("Press Enter to continue...")
    return True

def _view_file(path: Path) -> None:
    try:
        with open(path, 'r') as f:
            content = f.read()
        
        # Try to parse and pretty-print JSON to ensure formatting even if source is minified
        try:
            parsed = json.loads(content)
            content = json.dumps(parsed, indent=4)
        except json.JSONDecodeError:
            pass # Fallback to raw content if it's not valid JSON
            
        console.print(f"\n[bold magenta]--- Contents of {path.name} ---[/bold magenta]")
        syntax = Syntax(content, "json", theme="monokai", line_numbers=True)
        console.print(syntax)
        console.print("[bold magenta]-----------------------------------[/bold magenta]\n")
    except Exception as e:
        console.print(f"[red]Error reading file: {e}[/red]")
    input("Press Enter to return to the wizard...")

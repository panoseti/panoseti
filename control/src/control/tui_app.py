import json
import os
import typing
from pathlib import Path
from typing import Any, Type

from textual import work, on
from textual.app import App, ComposeResult
from textual.screen import Screen
from textual.widgets import (
    Header, Footer, ListView, ListItem, Label, Button, Input, Switch, Select, 
    Tree, Static, Tooltip
)
from textual.widgets.tree import TreeNode
from textual.containers import VerticalScroll, Horizontal, Vertical
from textual.message import Message

from pydantic import BaseModel, ValidationError
from pydantic.fields import FieldInfo

from control.utils.paths import PanoPaths
from control.utils.pydantic_config_models import (
    DataConfig, ObsConfig, DaqConfig, NetworkConfig, DaemonConfig, FirmwareConfig, QuaboUids, ImageMode, PulseHeightMode, WpsConfig
)

CONFIG_TYPES = {
    "data_config.json": DataConfig,
    "obs_config.json": ObsConfig,
    "daq_config.json": DaqConfig,
    "network_config.json": NetworkConfig,
    "daemon_config.json": DaemonConfig,
    "firmware_config.json": FirmwareConfig,
    "quabo_uids.json": QuaboUids,
}

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

def is_list_type(annotation: Any) -> bool:
    origin = typing.get_origin(annotation)
    if origin is list:
        return True
    if origin is not None:
        args = typing.get_args(annotation)
        for arg in args:
            if typing.get_origin(arg) is list:
                return True
    return False

class FieldWidgetUpdated(Message):
    """Fired when any input widget changes."""
    def __init__(self, path: tuple, value: Any):
        self.path = path
        self.value = value
        super().__init__()


class CustomInput(Input):
    def __init__(self, path: tuple, value: Any, field_type: Any, *args, **kwargs):
        super().__init__(str(value) if value is not None else "", *args, **kwargs)
        self.field_path = path
        self.field_type = field_type

    @on(Input.Changed)
    def handle_change(self, event: Input.Changed) -> None:
        val = event.value
        parsed_val = val
        if val == "":
            parsed_val = None
        else:
            import ast
            try:
                parsed_val = ast.literal_eval(val)
            except Exception:
                pass
        self.post_message(FieldWidgetUpdated(self.field_path, parsed_val))


class CustomSwitch(Switch):
    def __init__(self, path: tuple, value: bool, *args, **kwargs):
        super().__init__(value, *args, **kwargs)
        self.field_path = path

    @on(Switch.Changed)
    def handle_change(self, event: Switch.Changed) -> None:
        self.post_message(FieldWidgetUpdated(self.field_path, event.value))


class CustomSelect(Select):
    def __init__(self, path: tuple, options: list[tuple[str, Any]], value: Any, *args, **kwargs):
        super().__init__(options, value=value if value is not None else Select.BLANK, *args, **kwargs)
        self.field_path = path

    @on(Select.Changed)
    def handle_change(self, event: Select.Changed) -> None:
        val = event.value
        if val == Select.BLANK:
            val = None
        self.post_message(FieldWidgetUpdated(self.field_path, val))


class ErrorLabel(Label):
    """A label to display validation errors below a field."""
    pass


class EditorScreen(Screen):
    BINDINGS = [("escape", "app.pop_screen", "Back")]

    def __init__(self, file_path: Path, model_class: Type[BaseModel]):
        super().__init__()
        self.file_path = file_path
        self.model_class = model_class
        
        self.current_data = {}
        if self.file_path.exists():
            try:
                with open(self.file_path, 'r') as f:
                    self.current_data = json.load(f)
            except Exception:
                pass

        self.tree_data_map: dict[str, tuple[Type[BaseModel], tuple]] = {}
        self.active_path: tuple = ()
        self.error_labels: dict[tuple, ErrorLabel] = {}

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="editor_layout"):
            with VerticalScroll(id="nav_pane"):
                yield Label("Configuration Navigation", classes="pane_title")
                tree: Tree = Tree(self.file_path.name, data=("root", self.model_class, ()))
                tree.root.expand()
                self._populate_tree(tree.root, self.current_data, self.model_class, ())
                yield tree
                
                with Horizontal(classes="pane_actions"):
                    yield Button("Back", id="back_btn")
                    yield Button("Save", id="save_btn", variant="success")
                
            with VerticalScroll(id="form_pane"):
                yield Label("", id="global_errors", classes="hidden error_label")
                yield Label("Select a section on the left to edit.", id="form_title")
                yield Vertical(id="form_container")
                
        yield Footer()

    def _populate_tree(self, node: TreeNode, data: dict, model_class: Type[BaseModel], path: tuple):
        """Recursively build the tree from the Pydantic schema and current data."""
        # 1. Standard Fields
        for field_name, field_info in model_class.model_fields.items():
            if is_complex_type(field_info.annotation):
                field_path = path + (field_name,)
                val = data.get(field_name)
                
                if is_list_type(field_info.annotation):
                    list_node = node.add(f"{field_name} []", data=("list", field_info.annotation, field_path))
                    if isinstance(val, list):
                        inner_model = get_base_model(field_info.annotation)
                        for i, item in enumerate(val):
                            item_path = field_path + (i,)
                            item_node = list_node.add(f"[{i}]", data=("model", inner_model, item_path))
                            if inner_model and isinstance(item, dict):
                                self._populate_tree(item_node, item, inner_model, item_path)
                else:
                    inner_model = get_base_model(field_info.annotation)
                    if inner_model:
                        status = " (Disabled)" if val is None else ""
                        child_node = node.add(f"{field_name}{status}", data=("model", inner_model, field_path))
                        if isinstance(val, dict):
                            self._populate_tree(child_node, val, inner_model, field_path)

        # 2. Dynamic Extras (for extra='allow')
        if getattr(model_class, 'model_config', {}).get('extra') == 'allow':
            for k, v in data.items():
                if k not in model_class.model_fields:
                    if isinstance(v, dict) and get_base_model(model_class): # Heuristic for known custom blocks
                        # In reality we should check if k starts with image_ etc, but for true form we let them edit as a block
                        extra_path = path + (k,)
                        if k.startswith("image_"):
                            inner_model = ImageMode
                        elif k.startswith("pulse_height_"):
                            inner_model = PulseHeightMode
                        elif k.startswith("wps"):
                            inner_model = WpsConfig
                        else:
                            inner_model = None
                            
                        if inner_model:
                            child_node = node.add(f"{k} [Custom Mode]", data=("model", inner_model, extra_path))
                            self._populate_tree(child_node, v, inner_model, extra_path)

    @on(Tree.NodeSelected)
    def handle_tree_selection(self, event: Tree.NodeSelected) -> None:
        node_data = event.node.data
        if not node_data:
            return
            
        node_type, model_class, path = node_data
        self.active_path = path
        
        form_title = self.query_one("#form_title", Label)
        form_title.update(f"Editing: {' > '.join(str(p) for p in path) if path else 'Root'}")
        
        container = self.query_one("#form_container", Vertical)
        container.remove_children()
        self.error_labels.clear()

        # Resolve current sub-dictionary
        current_sub_data = self.current_data
        for p in path:
            if isinstance(current_sub_data, list) and isinstance(p, int):
                if p < len(current_sub_data):
                    current_sub_data = current_sub_data[p]
            elif isinstance(current_sub_data, dict):
                current_sub_data = current_sub_data.get(p, {})
                
        if current_sub_data is None:
            container.mount(Label("[This block is currently disabled/null]"))
            btn = Button("Enable Block", id="enable_block_btn", variant="primary")
            container.mount(btn)
            return

        if node_type == "list":
            container.mount(Label("List Manager"))
            btn = Button("[+] Add Item", id="add_list_item_btn", variant="primary")
            container.mount(btn)
            return

        # Render scalar fields for this model
        if model_class and hasattr(model_class, 'model_fields'):
            for field_name, field_info in model_class.model_fields.items():
                if not is_complex_type(field_info.annotation):
                    self._mount_scalar_field(container, field_name, field_info, current_sub_data.get(field_name), path + (field_name,))
            
            # Handle extras
            if getattr(model_class, 'model_config', {}).get('extra') == 'allow':
                container.mount(Static("--- Custom Fields ---", classes="section_divider"))
                for k, v in current_sub_data.items():
                    if k not in model_class.model_fields:
                        if not isinstance(v, dict) and not isinstance(v, list):
                            self._mount_scalar_field(container, k, None, v, path + (k,), is_extra=True)
                
                container.mount(Button("[+] Add Custom Extra Field/Block", id="add_extra_btn"))

        # Trigger an initial validation to highlight required fields if missing
        self._validate_form()

    def _mount_scalar_field(self, container: Vertical, name: str, field_info: FieldInfo | None, value: Any, path: tuple, is_extra: bool = False):
        with container.app.batch_update():
            header = Label(f"{name}{' (Extra)' if is_extra else ''}", classes="field_label")
            if field_info and field_info.description:
                header.tooltip = field_info.description
                
            container.mount(header)
            
            widget = None
            if field_info:
                ann = field_info.annotation
                origin = getattr(ann, '__origin__', None)
                args = getattr(ann, '__args__', ())
                
                if ann is bool or (origin is type(None) and bool in args) or (origin is typing.Union and bool in args):
                    widget = CustomSwitch(path, bool(value))
                elif origin is getattr(typing, 'Literal', None) or type(ann).__name__ == '_LiteralGenericAlias':
                    choices = [(str(a), a) for a in args]
                    if type(None) in args or origin is type(None):
                        choices.insert(0, ("None", Select.BLANK))
                    widget = CustomSelect(path, choices, value)
                else:
                    widget = CustomInput(path, value, ann)
            else:
                widget = CustomInput(path, value, str)
                
            container.mount(widget)
            
            error_label = ErrorLabel("", classes="error_label hidden")
            self.error_labels[path] = error_label
            container.mount(error_label)

    @on(Button.Pressed, "#enable_block_btn")
    def handle_enable_block(self, event: Button.Pressed) -> None:
        if not self.active_path: return
        parent_data = self.current_data
        for p in self.active_path[:-1]:
            parent_data = parent_data[p]
        last_key = self.active_path[-1]
        
        # Determine if it should be a dict or list
        tree: Tree = self.query_one(Tree)
        selected_node = tree.cursor_node
        node_type = selected_node.data[0] if selected_node and selected_node.data else "model"
        
        parent_data[last_key] = [] if node_type == "list" else {}
        self._rebuild_tree()
        self.handle_tree_selection(Tree.NodeSelected(selected_node))

    @on(Button.Pressed, "#add_list_item_btn")
    def handle_add_list_item(self, event: Button.Pressed) -> None:
        if not self.active_path: return
        parent_data = self.current_data
        for p in self.active_path:
            parent_data = parent_data.setdefault(p, [])
        parent_data.append({})
        self._rebuild_tree()
        
    @on(Button.Pressed, "#add_extra_btn")
    def handle_add_extra(self, event: Button.Pressed) -> None:
        # In a real app we'd prompt for key/type, for simplicity we add a dummy
        # Or we can notify the user to edit the JSON manually for complex block additions
        self.notify("To add a new block, please edit the JSON file directly or use the basic UI for now.", severity="warning")

    def _rebuild_tree(self):
        tree: Tree = self.query_one(Tree)
        tree.clear()
        tree.root.data = ("root", self.model_class, ())
        tree.root.label = self.file_path.name
        self._populate_tree(tree.root, self.current_data, self.model_class, ())
        tree.root.expand()

    @on(FieldWidgetUpdated)
    def handle_field_update(self, event: FieldWidgetUpdated) -> None:
        # Traverse and update current_data
        target = self.current_data
        path = event.path
        if not path: return
        
        for p in path[:-1]:
            if isinstance(target, dict):
                target = target.setdefault(p, {})
            elif isinstance(target, list) and isinstance(p, int):
                target = target[p]
                
        last_key = path[-1]
        if isinstance(target, dict):
            target[last_key] = event.value
        elif isinstance(target, list) and isinstance(last_key, int):
            target[last_key] = event.value
            
        self._validate_form()

    @work(exclusive=True, group="validate")
    async def _validate_form(self) -> None:
        # Debounce
        from asyncio import sleep
        await sleep(0.3)
        
        for el in self.error_labels.values():
            el.update("")
            el.add_class("hidden")
            
        global_label = self.query_one("#global_errors", Label)
        global_label.update("")
        global_label.add_class("hidden")
        
        try:
            self.model_class(**self.current_data)
        except ValidationError as e:
            global_errs = []
            for err in e.errors():
                # loc is a tuple of path components
                err_path = err['loc']
                if err_path in self.error_labels:
                    label = self.error_labels[err_path]
                    label.update(f"Error: {err['msg']}")
                    label.remove_class("hidden")
                else:
                    loc_str = " -> ".join([str(p) for p in err_path]) if err_path else "Root"
                    global_errs.append(f"  Field: {loc_str}\n  Error: {err['msg']}")
            
            if global_errs:
                global_label.update("Global/Unmapped Validation Errors:\n" + "\n".join(global_errs))
                global_label.remove_class("hidden")

    @on(Button.Pressed, "#back_btn")
    def handle_back(self, event: Button.Pressed) -> None:
        self.app.pop_screen()

    @on(Button.Pressed, "#save_btn")
    def handle_save(self, event: Button.Pressed) -> None:
        try:
            instance = self.model_class(**self.current_data)
            valid_data = instance.model_dump(exclude_none=True, by_alias=True)
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.file_path, 'w') as f:
                json.dump(valid_data, f, indent=4)
            self.notify("Saved successfully!", severity="information")
            self.app.pop_screen()
        except ValidationError as e:
            self.notify("Cannot save: Validation errors exist. Check the red labels in the form.", severity="error")


class WorkspaceScreen(Screen):
    BINDINGS = [("q", "app.quit", "Quit")]

    def compose(self) -> ComposeResult:
        yield Header()
        config_dir = PanoPaths.config_dir()
        with VerticalScroll():
            yield Label(f"PSETI Configuration Manager (TUI)", id="title")
            yield Label(f"Workspace: {config_dir}", id="subtitle")
            
            items = []
            for ct in CONFIG_TYPES.keys():
                path = config_dir / ct
                if path.is_symlink():
                    target = os.readlink(path)
                    if (path.parent / target).exists():
                        state = f"[yellow]Symlink -> {target}[/yellow]"
                    else:
                        state = f"[red]Broken Symlink -> {target}[/red]"
                elif path.exists():
                    state = "[green]Standalone[/green]"
                else:
                    state = "[dim]Missing[/dim]"
                    
                items.append(ListItem(Label(f"{ct}  -  {state}"), name=ct))
            yield ListView(*items, id="config_list")
        yield Footer()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        selected_type = event.item.name
        model_class = CONFIG_TYPES[selected_type]
        file_path = PanoPaths.config_dir() / selected_type
        
        if file_path.is_symlink():
            target = os.readlink(file_path)
            abs_target = file_path.parent / target
            if abs_target.exists():
                file_path = abs_target

        self.app.push_screen(EditorScreen(file_path, model_class))


class ConfigManagerApp(App):
    CSS = """
    #title { text-align: center; text-style: bold; color: cyan; margin-top: 1; }
    #subtitle { text-align: center; color: green; margin-bottom: 1; }
    #editor_layout { height: 1fr; }
    #nav_pane { width: 30%; border-right: solid green; padding: 1; }
    #form_pane { width: 70%; padding: 1 2; }
    .pane_title { text-style: bold; color: magenta; margin-bottom: 1; }
    .pane_actions { height: auto; margin-top: 1; }
    #form_title { text-style: bold; color: cyan; margin-bottom: 1; }
    .field_label { margin-top: 1; color: yellow; }
    .error_label { color: red; text-style: italic; }
    .hidden { display: none; }
    .section_divider { color: blue; margin-top: 2; }
    """
    def on_mount(self) -> None:
        self.push_screen(WorkspaceScreen())

if __name__ == "__main__":
    app = ConfigManagerApp()
    app.run()

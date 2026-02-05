"""
Plugin loader for mlcli.

Provides utilities for discovering and loading plugins.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Optional, Type, Union

from mlcli.plugins.base import Plugin


class PluginLoader:
    """
    Loader for discovering and managing plugins.
    
    Plugins can be loaded from:
    - Python packages (installed via pip)
    - Python files (loaded directly)
    - Entry points (discovered via setuptools)
    """
    
    def __init__(self) -> None:
        """Initialize plugin loader."""
        self._loaded_plugins: Dict[str, Plugin] = {}
        self._plugin_dirs: List[Path] = []
    
    def add_plugin_directory(self, path: Union[str, Path]) -> None:
        """
        Add a directory to search for plugins.
        
        Args:
            path: Path to plugin directory.
        """
        path = Path(path)
        if path.is_dir() and path not in self._plugin_dirs:
            self._plugin_dirs.append(path)
    
    def discover_from_entry_points(self, group: str = "mlcli.plugins") -> List[Type[Plugin]]:
        """
        Discover plugins from entry points.
        
        Args:
            group: Entry point group name.
        
        Returns:
            List of discovered plugin classes.
        """
        plugins = []
        
        try:
            from importlib.metadata import entry_points
            
            # Python 3.10+ or importlib_metadata
            eps = entry_points()
            if hasattr(eps, "select"):
                # Python 3.10+
                plugin_eps = eps.select(group=group)
            else:
                # Python 3.9
                plugin_eps = eps.get(group, [])
            
            for ep in plugin_eps:
                try:
                    plugin_cls = ep.load()
                    if isinstance(plugin_cls, type) and issubclass(plugin_cls, Plugin):
                        plugins.append(plugin_cls)
                except Exception as e:
                    print(f"Failed to load plugin from entry point {ep.name}: {e}")
        
        except ImportError:
            pass
        
        return plugins
    
    def discover_from_directory(self, path: Union[str, Path]) -> List[Type[Plugin]]:
        """
        Discover plugins from a directory.
        
        Args:
            path: Directory to search.
        
        Returns:
            List of discovered plugin classes.
        """
        path = Path(path)
        plugins = []
        
        if not path.is_dir():
            return plugins
        
        for file_path in path.glob("*.py"):
            if file_path.name.startswith("_"):
                continue
            
            try:
                plugin_cls = self._load_plugin_from_file(file_path)
                if plugin_cls:
                    plugins.append(plugin_cls)
            except Exception as e:
                print(f"Failed to load plugin from {file_path}: {e}")
        
        return plugins
    
    def _load_plugin_from_file(self, path: Path) -> Optional[Type[Plugin]]:
        """
        Load a plugin from a Python file.
        
        Args:
            path: Path to Python file.
        
        Returns:
            Plugin class or None.
        """
        module_name = f"mlcli_plugin_{path.stem}"
        
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            return None
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        
        # Look for Plugin subclass
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, Plugin)
                and attr is not Plugin
                and not attr.__name__.startswith("_")
            ):
                return attr
        
        return None
    
    def load_plugin(self, plugin: Union[str, Type[Plugin], Plugin]) -> Plugin:
        """
        Load and register a plugin.
        
        Args:
            plugin: Plugin class, instance, or module name.
        
        Returns:
            Loaded plugin instance.
        """
        if isinstance(plugin, str):
            # Load from module name
            module = importlib.import_module(plugin)
            
            # Find Plugin subclass in module
            plugin_cls = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, Plugin)
                    and attr is not Plugin
                ):
                    plugin_cls = attr
                    break
            
            if plugin_cls is None:
                raise ValueError(f"No Plugin class found in module: {plugin}")
            
            plugin = plugin_cls()
        
        elif isinstance(plugin, type) and issubclass(plugin, Plugin):
            plugin = plugin()
        
        if not isinstance(plugin, Plugin):
            raise TypeError(f"Expected Plugin instance, got: {type(plugin)}")
        
        # Register the plugin
        plugin.register()
        self._loaded_plugins[plugin.name] = plugin
        
        return plugin
    
    def unload_plugin(self, name: str) -> None:
        """
        Unload a plugin.
        
        Args:
            name: Plugin name.
        """
        if name in self._loaded_plugins:
            plugin = self._loaded_plugins[name]
            plugin.unregister()
            del self._loaded_plugins[name]
    
    def get_plugin(self, name: str) -> Optional[Plugin]:
        """
        Get a loaded plugin by name.
        
        Args:
            name: Plugin name.
        
        Returns:
            Plugin instance or None.
        """
        return self._loaded_plugins.get(name)
    
    def list_plugins(self) -> List[str]:
        """
        List all loaded plugins.
        
        Returns:
            List of plugin names.
        """
        return list(self._loaded_plugins.keys())
    
    def discover_all(self) -> List[Type[Plugin]]:
        """
        Discover all available plugins.
        
        Returns:
            List of discovered plugin classes.
        """
        plugins = []
        
        # From entry points
        plugins.extend(self.discover_from_entry_points())
        
        # From registered directories
        for directory in self._plugin_dirs:
            plugins.extend(self.discover_from_directory(directory))
        
        return plugins
    
    def load_all(self) -> List[Plugin]:
        """
        Discover and load all available plugins.
        
        Returns:
            List of loaded plugin instances.
        """
        loaded = []
        
        for plugin_cls in self.discover_all():
            try:
                plugin = self.load_plugin(plugin_cls)
                loaded.append(plugin)
            except Exception as e:
                print(f"Failed to load plugin {plugin_cls.__name__}: {e}")
        
        return loaded
    
    def unload_all(self) -> None:
        """Unload all loaded plugins."""
        for name in list(self._loaded_plugins.keys()):
            self.unload_plugin(name)


# Global plugin loader instance
_loader = PluginLoader()


def discover_plugins() -> List[Type[Plugin]]:
    """
    Discover all available plugins.
    
    Returns:
        List of discovered plugin classes.
    """
    return _loader.discover_all()


def load_plugin(plugin: Union[str, Type[Plugin], Plugin]) -> Plugin:
    """
    Load and register a plugin.
    
    Args:
        plugin: Plugin class, instance, or module name.
    
    Returns:
        Loaded plugin instance.
    """
    return _loader.load_plugin(plugin)


def get_plugin_loader() -> PluginLoader:
    """
    Get the global plugin loader.
    
    Returns:
        PluginLoader instance.
    """
    return _loader

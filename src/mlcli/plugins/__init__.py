"""Plugin system for mlcli."""

from mlcli.plugins.loader import PluginLoader, discover_plugins, load_plugin
from mlcli.plugins.base import Plugin, ModelPlugin, DatasetPlugin, TaskPlugin

__all__ = [
    "Plugin",
    "ModelPlugin",
    "DatasetPlugin",
    "TaskPlugin",
    "PluginLoader",
    "discover_plugins",
    "load_plugin",
]

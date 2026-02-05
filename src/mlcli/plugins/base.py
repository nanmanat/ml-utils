"""
Base plugin classes for mlcli.

Provides abstract base classes for creating plugins.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

import torch.nn as nn

from mlcli.core.registry import MODEL_REGISTRY, DATASET_REGISTRY, TASK_REGISTRY


class Plugin(ABC):
    """
    Base class for all plugins.
    
    Plugins can extend mlcli with new models, datasets, tasks,
    or other functionality.
    """
    
    name: str = "base_plugin"
    version: str = "0.1.0"
    description: str = "Base plugin"
    author: str = "Unknown"
    
    @abstractmethod
    def register(self) -> None:
        """
        Register the plugin's components.
        
        This method is called when the plugin is loaded.
        """
        pass
    
    def unregister(self) -> None:
        """
        Unregister the plugin's components.
        
        This method is called when the plugin is unloaded.
        Override if cleanup is needed.
        """
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get plugin information."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
        }


class ModelPlugin(Plugin):
    """
    Plugin for adding custom models.
    
    Subclass this to add new model architectures to mlcli.
    """
    
    def get_models(self) -> Dict[str, Type[nn.Module]]:
        """
        Get models provided by this plugin.
        
        Returns:
            Dictionary mapping model names to model classes.
        """
        return {}
    
    def register(self) -> None:
        """Register all models from this plugin."""
        for name, model_cls in self.get_models().items():
            MODEL_REGISTRY.register(name, model_cls)
    
    def unregister(self) -> None:
        """Unregister all models from this plugin."""
        for name in self.get_models().keys():
            MODEL_REGISTRY._registry.pop(name, None)


class DatasetPlugin(Plugin):
    """
    Plugin for adding custom datasets.
    
    Subclass this to add new dataset formats to mlcli.
    """
    
    def get_datasets(self) -> Dict[str, Type]:
        """
        Get datasets provided by this plugin.
        
        Returns:
            Dictionary mapping dataset names to dataset classes.
        """
        return {}
    
    def register(self) -> None:
        """Register all datasets from this plugin."""
        for name, dataset_cls in self.get_datasets().items():
            DATASET_REGISTRY.register(name, dataset_cls)
    
    def unregister(self) -> None:
        """Unregister all datasets from this plugin."""
        for name in self.get_datasets().keys():
            DATASET_REGISTRY._registry.pop(name, None)


class TaskPlugin(Plugin):
    """
    Plugin for adding custom tasks.
    
    Subclass this to add new task types to mlcli.
    """
    
    def get_tasks(self) -> Dict[str, Type]:
        """
        Get tasks provided by this plugin.
        
        Returns:
            Dictionary mapping task names to task classes.
        """
        return {}
    
    def register(self) -> None:
        """Register all tasks from this plugin."""
        for name, task_cls in self.get_tasks().items():
            TASK_REGISTRY.register(name, task_cls)
    
    def unregister(self) -> None:
        """Unregister all tasks from this plugin."""
        for name in self.get_tasks().keys():
            TASK_REGISTRY._registry.pop(name, None)


class CompositePlugin(Plugin):
    """
    Plugin that combines multiple plugins.
    
    Useful for creating plugin bundles.
    """
    
    def __init__(self, plugins: Optional[List[Plugin]] = None) -> None:
        """
        Initialize composite plugin.
        
        Args:
            plugins: List of plugins to combine.
        """
        self.plugins = plugins or []
    
    def add_plugin(self, plugin: Plugin) -> None:
        """Add a plugin to the composite."""
        self.plugins.append(plugin)
    
    def register(self) -> None:
        """Register all sub-plugins."""
        for plugin in self.plugins:
            plugin.register()
    
    def unregister(self) -> None:
        """Unregister all sub-plugins."""
        for plugin in self.plugins:
            plugin.unregister()
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about all sub-plugins."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "plugins": [p.get_info() for p in self.plugins],
        }

"""
Registry pattern for dynamic component registration.

Provides a thread-safe, type-hinted registry for models, datasets,
tasks, and other pluggable components.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Dict, Generic, List, Optional, Type, TypeVar, Union

T = TypeVar("T")


class RegistryError(Exception):
    """Exception raised for registry-related errors."""
    pass


class Registry(Generic[T]):
    """
    Generic registry for dynamic component registration.
    
    Supports decorator-based and explicit registration, aliasing,
    and hierarchical component lookup.
    
    Example:
        >>> model_registry = Registry[nn.Module]("models")
        >>> @model_registry.register("resnet50")
        ... class ResNet50(nn.Module):
        ...     pass
        >>> model_cls = model_registry.get("resnet50")
    """
    
    _instances: Dict[str, "Registry"] = {}
    _lock = threading.Lock()
    
    def __init__(self, name: str) -> None:
        """
        Initialize registry with a name.
        
        Args:
            name: Unique name for this registry instance.
        """
        self._name = name
        self._registry: Dict[str, Type[T]] = {}
        self._aliases: Dict[str, str] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
    
    @classmethod
    def get_instance(cls, name: str) -> "Registry":
        """
        Get or create a registry instance by name (singleton per name).
        
        Args:
            name: Registry name.
            
        Returns:
            Registry instance.
        """
        with cls._lock:
            if name not in cls._instances:
                cls._instances[name] = cls(name)
            return cls._instances[name]
    
    @property
    def name(self) -> str:
        """Get registry name."""
        return self._name
    
    def register(
        self,
        name: Optional[str] = None,
        aliases: Optional[List[str]] = None,
        **metadata: Any,
    ) -> Callable[[Type[T]], Type[T]]:
        """
        Decorator for registering a class.
        
        Args:
            name: Registration name (defaults to class name).
            aliases: Alternative names for lookup.
            **metadata: Additional metadata to store.
            
        Returns:
            Decorator function.
            
        Example:
            >>> @registry.register("my_model", aliases=["model_v1"])
            ... class MyModel:
            ...     pass
        """
        def decorator(cls: Type[T]) -> Type[T]:
            reg_name = name if name is not None else cls.__name__.lower()
            self.register_class(cls, reg_name, aliases=aliases, **metadata)
            return cls
        return decorator
    
    def register_class(
        self,
        cls: Type[T],
        name: str,
        aliases: Optional[List[str]] = None,
        force: bool = False,
        **metadata: Any,
    ) -> None:
        """
        Explicitly register a class.
        
        Args:
            cls: Class to register.
            name: Registration name.
            aliases: Alternative names.
            force: Overwrite existing registration.
            **metadata: Additional metadata.
            
        Raises:
            RegistryError: If name already registered and force=False.
        """
        with self._lock:
            name = name.lower()
            
            if name in self._registry and not force:
                raise RegistryError(
                    f"'{name}' already registered in {self._name} registry. "
                    f"Use force=True to overwrite."
                )
            
            self._registry[name] = cls
            self._metadata[name] = metadata
            
            # Register aliases
            if aliases:
                for alias in aliases:
                    alias = alias.lower()
                    if alias in self._aliases and not force:
                        raise RegistryError(
                            f"Alias '{alias}' already registered for '{self._aliases[alias]}'."
                        )
                    self._aliases[alias] = name
    
    def get(self, name: str) -> Type[T]:
        """
        Get a registered class by name or alias.
        
        Args:
            name: Registration name or alias.
            
        Returns:
            Registered class.
            
        Raises:
            RegistryError: If name not found.
        """
        name = name.lower()
        
        # Check direct registration
        if name in self._registry:
            return self._registry[name]
        
        # Check aliases
        if name in self._aliases:
            return self._registry[self._aliases[name]]
        
        # Name not found
        available = list(self._registry.keys()) + list(self._aliases.keys())
        raise RegistryError(
            f"'{name}' not found in {self._name} registry. "
            f"Available: {sorted(available)}"
        )
    
    def get_metadata(self, name: str) -> Dict[str, Any]:
        """Get metadata for a registered class."""
        name = name.lower()
        if name in self._aliases:
            name = self._aliases[name]
        return self._metadata.get(name, {})
    
    def contains(self, name: str) -> bool:
        """Check if name is registered."""
        name = name.lower()
        return name in self._registry or name in self._aliases
    
    def unregister(self, name: str) -> None:
        """Remove a registration."""
        with self._lock:
            name = name.lower()
            if name in self._registry:
                del self._registry[name]
                if name in self._metadata:
                    del self._metadata[name]
                # Remove associated aliases
                self._aliases = {k: v for k, v in self._aliases.items() if v != name}
            elif name in self._aliases:
                del self._aliases[name]
    
    def list_registered(self) -> List[str]:
        """List all registered names (excluding aliases)."""
        return sorted(self._registry.keys())
    
    def list_all(self) -> List[str]:
        """List all names including aliases."""
        return sorted(set(self._registry.keys()) | set(self._aliases.keys()))
    
    def clear(self) -> None:
        """Clear all registrations."""
        with self._lock:
            self._registry.clear()
            self._aliases.clear()
            self._metadata.clear()
    
    def __len__(self) -> int:
        """Return number of registered classes."""
        return len(self._registry)
    
    def __contains__(self, name: str) -> bool:
        """Check if name is registered."""
        return self.contains(name)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Registry(name='{self._name}', registered={len(self._registry)})"


# Global registry instances
MODEL_REGISTRY: Registry = Registry.get_instance("models")
DATASET_REGISTRY: Registry = Registry.get_instance("datasets")
TASK_REGISTRY: Registry = Registry.get_instance("tasks")
TRANSFORM_REGISTRY: Registry = Registry.get_instance("transforms")
METRIC_REGISTRY: Registry = Registry.get_instance("metrics")
OPTIMIZER_REGISTRY: Registry = Registry.get_instance("optimizers")
SCHEDULER_REGISTRY: Registry = Registry.get_instance("schedulers")


def register_model(
    name: Optional[str] = None,
    aliases: Optional[List[str]] = None,
    **metadata: Any,
) -> Callable[[Type[T]], Type[T]]:
    """Decorator for registering models."""
    return MODEL_REGISTRY.register(name, aliases, **metadata)


def register_dataset(
    name: Optional[str] = None,
    aliases: Optional[List[str]] = None,
    **metadata: Any,
) -> Callable[[Type[T]], Type[T]]:
    """Decorator for registering datasets."""
    return DATASET_REGISTRY.register(name, aliases, **metadata)


def register_task(
    name: Optional[str] = None,
    aliases: Optional[List[str]] = None,
    **metadata: Any,
) -> Callable[[Type[T]], Type[T]]:
    """Decorator for registering tasks."""
    return TASK_REGISTRY.register(name, aliases, **metadata)


def register_metric(
    name: Optional[str] = None,
    aliases: Optional[List[str]] = None,
    **metadata: Any,
) -> Callable[[Type[T]], Type[T]]:
    """Decorator for registering metrics."""
    return METRIC_REGISTRY.register(name, aliases, **metadata)

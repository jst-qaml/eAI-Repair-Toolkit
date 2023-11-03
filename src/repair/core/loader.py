"""Attribute Loader.

This file implements repair class loaders.
"""
from __future__ import annotations

import importlib
import pkgutil
from typing import TYPE_CHECKING

import repair.dataset
import repair.methods
import repair.model
import repair.utils
from repair.core.dataset import RepairDataset
from repair.core.method import RepairMethod
from repair.core.model import RepairModel

if TYPE_CHECKING:
    from types import ModuleType

    from repair.core._base import RepairClass


def _get_module_name(abspath: str) -> str:
    """Get module name.

    Examples
    --------
    >>> _get_module_name("repair.xxx.modname")
    "modname"

    Parameters
    ----------
    abspath : str
        Absolute package path

    Returns
    -------
    str
        Extracted module name

    """
    return abspath.split(".")[-1]


def _walk_namespace(ns: ModuleType) -> list[tuple[str, ModuleType, bool]]:
    """Walk plugin namespace.

    Get plugins and import them.

    Parameters
    ----------
    ns : ModuleType
        A target plugin namespace

    Returns
    -------
    avaiable_plugins : list[tuple[str, ModuleType, bool]]
        List of tuple that consists of name, module, ispkg

    """
    return [
        (_get_module_name(name), importlib.import_module(name), ispkg)
        for _, name, ispkg in pkgutil.iter_modules(ns.__path__, ns.__name__ + ".")
    ]


def _gather_repair_classes_from_pkg(pkg: ModuleType, kind: type[RepairClass]) -> dict[str, type]:
    """Gather repair classes from package.

    Parameters
    ----------
    pkg : ModuleType
        Target package
    kind : type[RepairClass]
        Kind of searching classes

    Returns
    -------
    dict[str, type]
        Dict of found repair classes. The key is its name and value is its class.

    """
    classes = {}
    all_attrs = vars(pkg).get("__all__", None)
    if all_attrs:
        for member in all_attrs:
            c = getattr(pkg, member, None)
            if c is None:
                continue

            if not isinstance(c, type):
                # skip if c is not a class
                continue

            if c is kind:
                # should not gather abstract base class itself
                continue

            if issubclass(c, kind):
                classes[c.get_name()] = c

    return classes


def _gather_repair_classes_from_module(
    module: ModuleType, kind: type[RepairClass]
) -> dict[str, type]:
    """Gather repair classes from module.

    Parameters
    ----------
    module : ModuleType
        Target module
    kind : type[RepairClass]
        Kind of searching classes

    Returns
    -------
    dict[str, type]
        Dict of found repair classes. The key is its name and value is its class.

    """
    classes = {}
    for member in vars(module):
        if member.startswith("__"):
            # ignore magic attributes
            continue

        c = getattr(module, member, None)
        if c is None:
            continue

        if not isinstance(c, type):
            # skip if c is not a class
            continue

        if c is kind:
            # should not gather abstract base class itself
            continue

        if issubclass(c, kind):
            classes[c.get_name()] = c

    return classes


def _gather_repair_classes(ns: ModuleType, kind: type[RepairClass]) -> dict[str, type]:
    """Import repair classes.

    Parameters
    ----------
    ns : str
        Root namespace to be walked.
    kind : type[RepairClass]
        Target class

    Returns
    -------
    Dict[str, type]
        Found repair classses

    Raises
    ------
    KeyError
        If multiple different repair classes have same name

    """
    available_classes = _walk_namespace(ns)
    classes = {}
    for _name, module, ispkg in available_classes:
        if ispkg:
            gathered = _gather_repair_classes_from_pkg(module, kind)
        else:
            gathered = _gather_repair_classes_from_module(module, kind)

        conflicted_keys = set(classes.keys()) & set(gathered.keys())
        if len(conflicted_keys) != 0:
            raise KeyError(f"Class name conflicted: {conflicted_keys}")
        classes.update(gathered)

    return classes


def load_repair_model(name: str) -> type[RepairModel]:
    """Load repair model.

    Parameters
    ----------
    name : str
        Target repair model name.
        This value should match with `get_name()` of target model class.

    Returns
    -------
    model : repair.core.RepairModel
        Found model class

    Raises
    ------
    KeyError
        If multiple different RepairModel have same name
    AttributeError
        If not found a model that matches with `name`

    """
    classes = _gather_repair_classes(repair.model, RepairModel)
    model = classes.get(name, None)
    if model is None:
        raise AttributeError(f"Model Not Found: {name}")

    return model


def load_repair_dataset(name: str) -> type[RepairDataset]:
    """Load repair dataset.

    Parameters
    ----------
    name : str
        Target repair dataset name.
        This value should match with `get_name()` of target dataset class.

    Returns
    -------
    dataset : repair.core.RepairDataset
        Found dataset class

    Raises
    ------
    KeyError
        If multiple different RepairDataset have same name
    AttributeError
        If not found a dataset that matches with `name`

    """
    classes = _gather_repair_classes(repair.dataset, RepairDataset)
    dataset = classes.get(name, None)
    if dataset is None:
        raise AttributeError(f"dataset Not Found: {name}")

    return dataset


def load_repair_method(name: str) -> type[RepairMethod]:
    """Load repair method.

    Parameters
    ----------
    name : str
        Target repair method name.
        This value should match with `get_name()` of target method class.

    Returns
    -------
    method : repair.core.RepairMethod
        Found method class

    Raises
    ------
    KeyError
        If multiple different Repairmethod have same name
    AttributeError
        If not found a method that matches with `name`

    """
    classes = _gather_repair_classes(repair.methods, RepairMethod)
    method = classes.get(name, None)
    if method is None:
        raise AttributeError(f"method Not Found: {name}")

    return method


def load_utils(name: str) -> ModuleType:
    """Load utility module.

    Parameters
    ----------
    name : str
        Name of utility module.

    Returns
    -------
    ModuleType
        Loaded utility module

    Raises
    ------
    ModuleNotFoundError
        If target utility is not found.
    TypeError
        If loaded modules doesn't satisfy spec.

    """
    util = importlib.import_module(f".{name}", repair.utils.__name__)
    if not hasattr(util, "run"):
        raise TypeError(f"Does not have entrypoint: {name}")

    return util

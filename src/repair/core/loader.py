"""Attribute Loader.

This file implements repair class loaders.
"""
from __future__ import annotations

import importlib
import pkgutil
from typing import TYPE_CHECKING

from repair.core.dataset import RepairDataset
from repair.core.exceptions import (
    RepairDatasetNotFoundError,
    RepairError,
    RepairMethodNotFoundError,
    RepairModelNotFoundError,
    RepairModuleError,
    RepairUtilNotFoundError,
)
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


def _get_namespace_package_path(nsmod_name: str) -> list[str]:
    """Get package path from given namespace string.

    Parameters
    ----------
    nsmod_name: str
        A target namespace package

    Returns
    -------
    list[str]
        List of path-strings to given namespace package

    Raises
    ------
    RepairModuleError
        When trying to import invalid namespace, directory structure itself is invalid and so on.
    RepairError
        When unreachable block has been reached. You should report to developer ASAP if you meet
        this error.

    """
    _allowed_namespaces = [
        "repair.methods",
        "repair.dataset",
        "repair.model",
        "repair.utils",
    ]
    if nsmod_name not in _allowed_namespaces:  # pragma: no cover
        # Exclude this branch from coverage because we cannot run repair framework
        # in this situation anyway.
        errmsg = "Trying to import modules from invalid namespace."
        raise RepairModuleError(errmsg)

    try:
        nsmodule = importlib.import_module(nsmod_name)
    except ImportError as ie:  # pragma: no cover
        # Exclude this branch from coverage because we cannot run repair framework
        # in this situation anyway.
        errmsg = f"{nsmod_name} does not exist on the path."
        raise RepairModuleError(errmsg) from ie

    # We can assume `nsmodule` always has `__file__` attribute
    # because we limit the acceptable namespace above.
    if nsmodule.__file__ is not None:  # pragma: no cover
        # Exclude this branch from coverage because we cannot run repair framework
        # in this situation anyway.
        errmsg = f"""'{nsmod_name}' is not native namespace.
NOTE: You should remove '__init__.py' if it exists at '{nsmodule.__file__}'."""
        raise RepairModuleError(errmsg)

    if hasattr(nsmodule, "__path__"):
        return list(nsmodule.__path__)
    else:  # pragma: no cover
        # Packages always have `__path__` attribute.
        # If `__path__` does not exist, it means `nsmodule` is not a package.
        # See https://docs.python.org/ja/3/reference/import.html#path__ for more detail.
        raise RepairError("FATAL: Unreachable block has been reached!")


def _walk_namespace(ns: str) -> list[tuple[str, ModuleType, bool]]:
    """Walk plugin namespace.

    Get plugins and import them.

    Parameters
    ----------
    ns : str
        A target plugin namespace

    Returns
    -------
    avaiable_plugins : list[tuple[str, ModuleType, bool]]
        List of tuple that consists of name, module, ispkg

    """
    nspath = _get_namespace_package_path(ns)
    return [
        (_get_module_name(name), importlib.import_module(name), ispkg)
        for _, name, ispkg in pkgutil.iter_modules(nspath, ns + ".")
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


def _gather_repair_classes(ns: str, kind: type[RepairClass]) -> dict[str, type]:
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
    RepairModuleError
        If multiple different repair classes have the same name

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
            raise RepairModuleError(f"Class name conflicted: {conflicted_keys}")
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
    RepairModuleError
        If multiple different RepairModel have the same name
    RepairModelNotFoundError
        If not found a model that matches with `name`

    """
    classes = _gather_repair_classes("repair.model", RepairModel)
    model = classes.get(name, None)
    if model is None:
        raise RepairModelNotFoundError(name)

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
    RepairModuleError
        If multiple different RepairDataset have the same name
    RepairDatasetNotFoundError
        If not found a dataset that matches with `name`

    """
    classes = _gather_repair_classes("repair.dataset", RepairDataset)
    dataset = classes.get(name, None)
    if dataset is None:
        raise RepairDatasetNotFoundError(name)

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
    RepairModuleError
        If multiple different RepairMethod have the same name
    RepairMethodNotFoundError
        If not found a method that matches with `name`

    """
    classes = _gather_repair_classes("repair.methods", RepairMethod)
    method = classes.get(name, None)
    if method is None:
        raise RepairMethodNotFoundError(name)

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
    RepairUtilNotFoundError
        If target utility is not found.
    RepairModuleError
        If loaded modules doesn't satisfy spec.

    """
    try:
        util = importlib.import_module(f".{name}", "repair.utils")
    except ImportError as ie:
        raise RepairUtilNotFoundError(name) from ie

    if not hasattr(util, "run"):
        raise RepairError(f"Util function does not have entrypoint: {name}")

    return util

"""Repair exception classes."""


class RepairError(Exception):
    """Base exception of repair related exceptions."""

    pass


class RepairModuleError(RepairError):
    """General exception of model/dataset/method/utils related operation."""

    pass


class BaseModuleNotFoundError(RepairError):
    """Base exception for some repair related module are not found."""

    def __init__(self, module_name: str):
        message = f"{module_name} is not available."
        super().__init__(message)


class RepairMethodNotFoundError(BaseModuleNotFoundError):
    """Exception raised for trying to unknown methods."""

    def __init__(self, method_name: str):
        super().__init__(method_name)


class RepairDatasetNotFoundError(BaseModuleNotFoundError):
    """Exception raised for trying to unknown datasets."""

    def __init__(self, dataset_name: str):
        super().__init__(dataset_name)


class RepairModelNotFoundError(BaseModuleNotFoundError):
    """Exception raised for trying to unknown models."""

    def __init__(self, model_name: str):
        super().__init__(model_name)


class RepairUtilNotFoundError(BaseModuleNotFoundError):
    """Exception raised for tryping to unknown utils."""

    def __init__(self, util_name: str):
        super().__init__(util_name)


class RepairModelError(RepairError):
    """Exception about RepairModel."""

    pass

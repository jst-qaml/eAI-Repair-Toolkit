from abc import ABC, abstractmethod


class RepairClass(ABC):
    """Root abstract class of all Repair classes."""

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """Returns name of this class."""
        pass

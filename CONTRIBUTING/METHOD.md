# Contributing Guidelines for Method Implementers

When you want to add new repair methods, follow the steps below.

## Creating Modules or Packages

We adopt [src-layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/) to manage this project.
Create modules or packages under `src/repair/methods` for your repair methods.
If you make new repair methods as packages, you should put `__init__.py` under `src/repair/methods/<your method>`.
A file structure should be like below after you put file(s).

```
src/
    repair/
        methods/
            your_method_module.py # when you implement as module
            your_method_pkg/ # when you implement as package
                __init__.py
                file1.py
                file2.py
                ...
```

Be sure **NOT** to put `__init__.py` at `src/repair/__init__.py` and `src/repair/methods/__init__.py`!
This will break the Python's namespace functionality we expect.

## Implementimg Method Classes

Define classes of your repair methods and inherit the `RepairMethod` base class.

```python
from repair.core.method import RepairMethod 


class YourMethod(RepairMethod):
    ...

```

To implement the class methods below, read [`src/repair/core/method.py`](../src/repair/core/method.py).

### (optional) get_name

This function returns the name that is used for loading your method from the `src/repair/cli` module.
A default value is a class name of your method.
You can override this function to return a name you want.

Example:
```python
class YourMethod(RepairMethod):

    @classmethod
    def get_name(cls) -> str:
        # default value is `YourMethod`
        return "my_method"

```

In this example, you can call this method from CLI like `repair localize --method=my_method ...`.

### localize

This function is called for fault localization of target models.
The localized weights will be saved in the `target_data_dir` because the `optimize` function read it from there.
You also should display the result. The degree of the verbosity should be controlled by the option `verbose`.
Parameters are as below.

- `model`: Target model read from `model_dir`
- `input_neg`: Dataset read from `target_data_dir`.
- `output_dir`: Path to a directory designated by `target_data_dir`.
- `verbose`: Degree of the output verbosity designated by the option `verbose`.

### optimize

This function is called for modifying the weight value localized in the `localized` function.
You also should display the result. The degree of the verbosity should be controlled by the option `verbose`. Parameters are as below.

- `model`: Target model read from `model_dir`
- `model_dir`: Path to a directory containing model
- `weights`: Localized weight values read from `target_data_dir`
- `input_neg`: Dataset read from `target_data_dir`.
- `input_pos`: Dataset read from `positive_inputs_dir`.
- `output_dir`: Path to a directory designated by `output_dir`.
- `verbose`: Degree of verbosity designated by the option `verbose`.

### evaluate

This function is called when experimenting some dataset with this repair method.
This function usually calls the `localize` function and the `optimize` function to get results. You also should display the result.
The degree of the verbosity should be controlled by the option `verbose`. Parameters are as below.
  
- `dataset`: RepairDataset module designated by the option `dataset`.
- `model_dir`: Path to a directory designated by the option `model_dir`.
- `target_data`: Dataset read from `target_data_dir`.
- `target_data_dir`: Path to a directory designated by the option `target_data_dir`.
- `positive_inputs`: Dataset read from `positive_inputs_dir`.
- `positive_inputs_dir`: Path to a directory designated by the option `positive_inputs_dir`.
- `output_dir`: Path to a directory designated by the option `output_dir`.
- `num_runs`: The number of executions designated by the option `num_runs`.
- `verbose`: Degree of verbosity designated by the option `verbose`.

### save_weights

Saving weights localized by the `localize` function. This function is not called from elsewhere. You should call this function at the end of `localize`. Parameters are as below.

- `weights`: Weight values calculated by the `localize` function.
- `output_dir`: Path to a directory where the weights are saved. This path should be the same as the `target_data_dir` because the `load_weights` function loads the weights from there.

### load_weights

Loading weights localized by the `localize` function. This function is called by `optimize` in `cli.py`. Parameters are as below.

- `output_dir`: Path to a directory designated by the option  `positive_inputs_dir`

### load_input_neg

Loading a negative dataset created by `target` function of `dataset` module. This function is called by `localize` and `optimize` in `cli.py`. Parameters are as below.

- `neg_dir`: Path to a directory designated by the option  `target_data_dir`

### load_input_pos

Loading a positive dataset created by `target` function of `dataset` module. This function is called by `optimize` in `cli.py`.  Parameters are as below.

- `pos_dir`: Path to a directory designated by the option  `positive_inputs_dir`

### set_options

If you want to add new options, you can implement here. For example, if you add a new option `new_option`, you can get the value of `new_option` from `kwargs` as below.

```python
def set_options(self, **kwargs):

    if 'new_option' in kwargs:
        self.new_option = kwargs['new_option']

```

### (optional) Define `__all__`

We recommend you to define `__all__` at `__init__.py` if you develop your method as package to make it available via `repair` cli.
You can define `__all__` like example below:

```python
# at your_method_pacakge/__init__.py
from .your_files import YourMethod, YourMethod2

__all__ = ["YourMethod", "YourMethod2", ...]

```

See the official docs to reference about [`__all__`](https://docs.python.org/3/tutorial/modules.html?highlight=import#packages)

## Updating Documents

You need to update the [Section Automated Repair](../docs/user_manual#5-automated-repair) in the user manual.
Add the name of your repair method and its brief description in the Table Repair Methods.
In addition, update the Tables Options in the subsections of localizing and optimizing according to your implementation,
especially about method-specific parameters configured by `set_options(**kwargs)`.

If you modify the base class and add/modify/remove the class methods in [`src/repair/core/method.py`](../src/repair/core/method.py), update this document besides the user manual.

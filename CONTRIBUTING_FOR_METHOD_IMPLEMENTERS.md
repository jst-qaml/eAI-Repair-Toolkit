# Contributing guidelines for method implementers

This guideline is for the developers who would like to implement new repair methods independent from eAI-Repair-Toolkit repository.

## LICENSE of implementing methods

You can set any license to your repair methods under your responsibility as long as you provide it in independent repository. 


## Python Runtime

### Version

You should support python version compatible with this project. This project supports versions that are officialy supported.

### Package Manager

You may use any python package managers such as pip, [poetry](https://python-poetry.org/), [pdm](https://pdm.fming.dev/latest/) and so on.

Note: We do NOT recommend to use [conda](https://docs.conda.io/projects/conda/en/stable/#) with pip because the problems may occur due to difference between conda and pip in the way to manage dependencies. If you want to use conda, use it at your own risks.

## Implementing a new method

### Make module or package

We adopt [src-layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/) to manage this project.
You should place your method module or package under `src/repair/methods`.
If you make new method as package, you should place `__init__.py` under `src/repair/methods/<your method>`.
File structure should be like below after you place file(s).

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

Be sure **NOT** to place `__init__.py` at `repair/__init__.py` and `repair/methods/__init__.py`!
This will break python's namespace functionality we expect.

### Implement method class
Define your method class and inherit `RepairMethod` base class.

```python
from repair.core.method import RepairMethod 


class YourMethod(RepairMethod):
    ...

```

Implement methods below:

### (optional) get_name

This function returns the name that is used to load your method from repair cli.
Default value is class name of your method.
You can override this function to return the name you want to provide.

```python
class YourMethod(RepairMethod):
    @classmethod
    def get_name(cls) -> str:
        # default value is `YourMethod`
        return "my_method"

```

In this example, you can load this method from cli like `repair localize --method=my_method ...`.

### localize

This function is called for fault localization of the target model.
The localized weights should be saved in the `target_data_dir` because the `optimize` function read it from there.
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
  - `dir`: Path to a directory designated by the option  `target_data_dir`

### load_input_pos

  Loading a negative dataset created by `target` function of `dataset` module. This function is called by `optimize` in `cli.py`.  Parameters are as below.
  - `dir`: Path to a directory designated by the option  `positive_inputs_dir`

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

See official docs to reference about `__all__` (https://docs.python.org/3/tutorial/modules.html?highlight=import#packages)


### Provide documents

We recommend to provide the document for your new method. In specially, you should describe method-specific parameters configured by `set_options(**kwargs)`.

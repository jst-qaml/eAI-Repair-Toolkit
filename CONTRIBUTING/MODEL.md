# Contributing Guidelines for Model Implementers

When you want to add new models, follow the steps below.

## Creating Modules or Packages

We adopt [src-layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/) to manage this project.
Create modules or packages under `src/repair/model` for your model.
If you make new model as package, you should put `__init__.py` under `src/repair/model/<your model>`.
A file structure should be like below after you put file(s).

```
src/
    repair/
        model/
            your_model_module.py # when you implement as module
            your_model_pkg/ # when you implement as package
                __init__.py
                file1.py
                file2.py
                ...
```

Be sure **NOT** to place `__init__.py` at `repair/__init__.py` and `repair/model/__init__.py`!
This will break python's namespace functionality we expect.

### Implementing Model Classes

Define classes of your model and inherit `RepairModel` base class.

```python
from repair.core.model import RepairModel 


class YourModel(RepairModel):
    ...

```

To implement the class methods below, read [`src/repair/core/model.py`](../src/repair/core/model.py).

### get_name

This function returns name that is used to load your model from the `src/repair/cli` module.
A default value is a class name of your model.
You can override this function to return a name you want.

Example:
```python
class YourModel(RepairModel):

    @classmethod
    def get_name(cls) -> str:
        # default value is `YourModel`
        return "my_model"

```

In this example, you can call this model from CLI like `repair train --model=my_model ...`.

### compile

This function is called for compiling a model in `train` command.
The model is thought to be created by using keras [sequential model](https://www.tensorflow.org/guide/keras/sequential_model)
or [function API](https://www.tensorflow.org/guide/keras/functional).
Parameters are as below.

- `input_shape`: Shape of model inputs, that is, the size of input images for training
- `output_shape`: Shape of model outputs, that is, the number of classes of a dataset for training.

### set_extra_config

If you want to add any extra configs, you can implement here.
For example, if you add a new option `new_option`, the value of it can be gotten from `kwargs` as below.
This function is called after generating this object in `cli.py`.

```python
  def set_extra_config(self, **kwargs):
      if 'new_option' in kwargs:
          self.new_option = kwargs['new_option']

  ```

### (optional) Define `__all__`

You should define `__all__` at `__init__.py` if you develop your model as package to make it available via cli.
You can define `__all__` like example below:

```python
# at your_model_pacakge/__init__.py
from .your_files import YourModel, YourModel2

__all__ = ["YourModel", "YourModel2", ...]

```

See the official docs to reference about [`__all__`](https://docs.python.org/3/tutorial/modules.html?highlight=import#packages)

## Updating Documents

You need to update the [Section DNN Model](../docs/user_manual#3-dnn-model) in the user manual.
Add the name of your model to the Table Options in the [Subsection Training](../docs/user_manual#31-training).
Also add new options if you implement.

If you modify the base class and add/modify/remove the class methods in [`src/repair/core/model.py`](../src/repair/core/model.py), update this document besides the user manual.

# Contributing Guidelines for Dataset Implementers

When you want to add new datasets, follow the steps below.

## Creating Modules or Packages

We adopt [src-layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/) to manage this project.
Create modules or packages under `src/repair/dataset` for your datasets.
If you make new dataset as packages, you should put `__init__.py` under `src/repair/dataset/<your dataset>`.
A file structure should be like below after you put file(s).

```
src/
    repair/
        dataset/
            your_dataset_module.py # when you implement as module
            your_dataset_pkg/ # when you implement as package
                __init__.py
                file1.py
                file2.py
                ...
```

Be sure **NOT** to create `__init__.py` at `src/repair/__init__.py` and `src/repair/dataset/__init__.py`!
This will break the Python's namespace functionality we expect.

## Implementing Dataset Classes

Define classes of your dataset and inherit `RepairDataset` base class.

```python
from repair.core.dataset import RepairDataset 


class YourDataset(RepairDataset):
    ...

```

To implement the class methods below, read [`src/repair/core/dataset.py`](../src/repair/core/dataset.py).

### get_name

This function returns name that is used to load your dataset from the `src/repair/cli` module.
A default value is a class name of your dataset.
You can override this function to return a name you want.

Example:
```python
class YourDataset(RepairDataset):

    @classmethod
    def get_name(cls) -> str:
        # default value is `YourDataset`
        return "my_dataset"

```

In this example, you can call this dataset from CLI like `repair prepare --dataset=my_dataset`.

### get_label_map

This function returns the map containing label names and corresponding integers.
This return value must be Python's `dict`: its key and value are the label name and the corresponding integer.
This method must be declared as a static method so that it can be called from outside of the class or from an its instance method.

Example:
```python
class YourDataset(RepairDataset):

    @staticmethod
    def get_label_map() -> dict[str, int]:
        return {
            "Apple": 1,
            "Orange": 2,
            "Banana": 3,
        }

```

### prepare

This function takes raw image files and outputs integrated dataset files.
This function should divide a dataset into 3: for training, testing, and repairing.
You must not share any image data over the 3 dataset to prevent data leak.
The output is a three dataset files.
The file format standard is as below. 

- Dataset files are created by calling `repair.core.dataset.RepairDataset.save_dataset_as_hdf`
- The image data is stored in `numpy.array` with `dtype='float32'`
- The label data is stored in `numpy.array` with `dtype='unit8'`

If you save the dataset file with the following format, you can omit implementing any other functions in the `dataset` module.
Parameters are as below.

- `input_dir`: Path to a directory designated by `input_dir`.
- `output_dir`: Path to a directory designated by `output_dir`.
- `divide_rate`: Used for a rate of data dividing designated by `divide_rate`.
- `random_state`: Used for a seed value for sampling dividing designated by `random_state`.


### load_test_data

This function loads a data for testing. The outputs are images and labels for testing.
You can omit implementing this function if you follow the file format standard.
Parameters are as below.

- `data_dir`: Path to a directory designated by `data_dir`.
  
### load_repair_data

This function loads a data for repairing. The outputs are images and labels for repairing.
You can omit implementing this function if you follow the file format standard.
Parameters are as below.

- `data_dir`: Path to a directory designated by `data_dir`.

### set_extra_config

If you want to add any extra configs, you can implement here.
For example, if you add a new option `new_option`, you can get the value of `new_option` from `kwargs` as below.
This function is called after generating this object in `cli.py`.

```python
  def set_extra_config(self, **kwargs):
      if 'new_option' in kwargs:
          self.new_option = kwargs['new_option']

  ```

### (optional) `__all__`

You should define `__all__` at `__init__.py` if you develop your dataset as package to make it available via cli.
You can define `__all__` like the example below:

```python
# at your_dataset_pacakge/__init__.py
from .your_files import YourDataset, YourDataset2, ...

__all__ = ["YourDataset", "YourDataset2", ...]

```

See the official docs to reference about [`__all__`](https://docs.python.org/3/tutorial/modules.html?highlight=import#packages)

## Updating Documents

You need to update the [Section Dataset](../docs/user_manual#2-dataset) in the user manual.
Create a subsection and add a brief description about your dataset and how to get actual data.
In addition, add the name of your dataset to the Table Options in the [Section Preparation](../docs/user_manual#26-preparation).
Also add new options if you implement.

If you modify the base class and add/modify/remove the class methods in [`src/repair/core/dataset.py`](../src/repair/core/dataset.py), update this document besides the user manual.

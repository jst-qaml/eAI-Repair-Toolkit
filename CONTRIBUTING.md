# Contributing to eAI-Repair

The following is a set of guidelines for contributing to eAI-Repair development, which are hosted in the [JST-QAML](https://github.com/jst-qaml) on GitHub. These are mostly guidelines, not rules. Use your best judgment, and feel free to propose changes to this document in a pull request.


#### Table Of Contents

[How Can I Contribute?](#how-can-i-contribute)
  * [Creating an Issue](#creating-an-issue)
  * [Creating a Branch](#creating-a-branch)
  * [Pull Requests](#pull-requests)

[How Can I Change the Code?](#how-can-i-change-the-code)
  * [Adding a new dataset](#adding-a-new-dataset)
  * [Adding a new model](#adding-a-new-model)
  * [Adding a new repair method](#adding-a-new-repair-method)
  * [Code Formatting](#code-formatting)
  * [Linting](#linting)
  * [Pre-commit](#pre-commit)


## How Can I Contribute?

This repository applies issue centric development. Before fixing bugs, adding a new feature or any other changing to this repository, you should create a related issue at first. Then you create a branch related to the issue and create a pull request with the branch to close the issue.

### Creating an Issue

- Reporting Bugs

  If you find bugs, please report it with the template as below. You should write `Overview`, `How to Reproduce` and `Expected Behavior` to make developers understand what happens precisely. The description of the way of reproducing the bug would be better as minimal as possible.  Attach a helpful `Screenshots` if you have. Please write any other information, like your idea for causing the bug or fixing it, in `Miscellaneous`.

  <details><summary> Show template </summary>

  ```
  ## Overview
  
  Describe a target bug clearly and briefly.
  
  ## How to Reproduce
  
  For example, 
  
  1. Open '...' menu
  2. Select '....'
  3. Move to '....'
  4. Find errors
  
  ## Expected Behavior
  
  Describe expected behavior clearly and briefly.
  
  ## Screenshots
  
  Provide hints to facilitate understanding this issue, if possible.
  
  ## Miscellaneous
  
  ```

  </details>

- Requesting a New Feature

  If you want a new feature, please request it with the template as below. You should write `Request` at least to let developers know what you want. It would be better to write `as is` and `to be`, or background of the request in addition to the request itself. Write your implementing idea for the request in `How to Implement` or `Alternatives` if you have. 

  <details><summary> Show template</summary>

  ```
  ## Request
  
  Describe new features and improvement you want.
  
  ## How to Implement
  
  Describe ideas to realize the features and improvement.
  
  ## Alternatives
  
  If any, describe ideas other than the above.
  
  ## Miscellaneous
  
  ```

  </details>

- Asking a Question

  If you have a question about this repository, ask it with the template as below. 

  <details><summary> Show template</summary>

  ```
  ## Question

  Describe insightful troubles you have.

  ## Miscellaneous

  ```

  </details>

- Maintenancing Project and Repository 

  If you find some typos, request to manage packages used in this project, or request to configure some developer tools, ask it with the template as below.

  <details><summary> Show template</summary>

  ```
  ## Description

  Describe typos you found, what you request to add/update/remove packages or what you want to configure tools here.


  ## Miscellaneous
  ```

  </details>

- Refactoring Code

  If you want to change code to improve performance, maintenability etc., ask it with the template as below.

  <details><summary> Show template</summary>

  ```
  ---
  name: Refactor
  about: refactoring codes
  title: ''
  labels: 'Type: Refactoring'
  assignees: ''

  ---

  ## Description

  Describe the purpose of refactoring you will do, like improving performance, maintenability etc.
  You can also describe more detailed extra information to AsIs and ToBe section.


  ## As Is


  ## To Be


  ```

  </details>

Also you can create other types of issues without using the templates above.

### Creating a Branch

Before creating a branch, you should create an issue related to it. Developers want to know what is proceeding in the repository. Therefore, the name of branch should contains the issue number and the class of changes like `feat`, `fix`, `docs`, `refactor`, `chore` and `tests`. 

- feat: Creating new features and improvement.
- fix: Fixing bugs.
- docs: Modifying or creating a document.
- refactor: Refactoring the codes.
- chore: Other changes like re-configuring virtual environment.
- tests: Create or add tests

For example, the branch named `feat-70` is a branch that adding a new feature about issue #70. You can also attach more concrete contents after the branch name  (like `feat-70-add-hydranet-model`) .

### Pull Requests

Before creating a pull request, you should create an issue related to it. Reviewers want to know why this pull request has created, and these kinds of information should be stored in issues. Pull requests should be created based on the template as below. 
You should write an overview of what you do and the issue number to be closed in `Details`. Attaching screenshots or short gif files would be helpful to make reviewers understand the pull request.
Before creating the pull request, you have to do tests!  Write what you tested with respect to your changes in `Tests`. The rules for test files are as below.
- The test files should placed in `tests` directory. 
- The name of test files should start with `test_`.
- The name of test classes should end with `Test`.

Then, you should modify the PR title to keep semantic. You should denote the type of PR at the head of title.

Template:

```
type(scope): title
```

| keys | description | remarks |
| :--: | :---------- | :-----: |
| type | Name of PR type. Acceptable types are same as kinds listed in [Creating a Branch](### Creating a Branch). | Required |
| scope | You can also denote the scope of impact of your PR. Name of scope should be the name of modules or directories. | Optional |
| title | Main title of your PR | Required |

Examples:

```
# without scope
feat: add new repair model
docs: fix typo in docs

# with scope
fix(repair): add missing options for arachne
tests: add tests for athena
```

#### Available Scopes

You can use the terms below as PR scope. If you have trouble choosing a scope, you can omit it. Or consider restructuring your PR so that you can choose the scope.
You are also welcome to suggest to add extra scopes.

- dataset
- demo
- docs
- model
- method
- tests

At last, You should check the `Check List` to ensure the quality of this repository. Of course, `pytest`, `pylama` and `isort`  should be run and all the warnings should be corrected.

<details><summary> Show template</summary>

```

## Details

Clearly and briefly describe details of changes and linked issue. If any, describe contexts and dependencies related to the changes.

Close # (linked issue)

## Tests

List test items. If any, describe test configuration. Any materials such as screenshots to facilitate to understanding the test items are welcome.

- [ ] Test A
- [ ] Test B

## Check List

- [ ] We have reviewed by our own.
- [ ] We have written suitable comments.
- [ ] We have modified related parts of documents.
- [ ] We have confirmed no new warning by the changes.
    - [ ] `pytest`
    - [ ] `ruff check .`
    - [ ] `black .`
- [ ] We have included all dependencies.

```

</details>


## How Can I Change the Code

### Adding a new dataset

#### Implementing a new dataset

When you want to add new dataset, follow the steps below.

##### Make module or packge
Place your dataset module or package under `src/repair/dataset`.
If you make new dataset as package, you should place `__init__.py` under `src/repair/dataset/<your dataset>`.
File structure should be like below after you place file(s).

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

Be sure **NOT** to place `__init__.py` at `repair/__init__.py` and `repair/dataset/__init__.py`!
This will break python's namespace functionality we expect.

##### Implement dataset class
Define your dataset class and inherit `RepairDataset` base class.

```python
from repair.core.dataset import RepairDataset 


class YourDataset(RepairDataset):
    ...

```

Implement methods below:

##### get_name
This function returns name that is used to load your dataset from repair cli.
Default value is class name of your dataset.
You can override this function to return the name you want to provide.

Example:
```python
class YourDataset(RepairDataset):
    @classmethod
    def get_name(cls) -> str:
        # default value is `YourDataset`
        return "my_dataset"

```
In this example, you can load this dataset in cli like `repair prepare --dataset=my_dataset`.

##### prepare
  This function takes raw image files and outputs integrated dataset files. This function should divide a dataset into 3: for training, testing, and repairing. You must not share any image data over the 3 dataset to prevent data leak. The output is a three dataset files. The file format standard is as below. 
  - Dataset files are created by calling `repair.core.dataset.Dataset.save_dataset_as_hdf`
  - The image data is stored in `numpy.array`  with `dtype='float32'`
  - The label data is stored in `numpy.array` with `dtype='unit8'`

  If you save the dataset file with the following format, you can omit implementing any other functions in the `dataset` module. Parameters are as below.
  - `input_dir`: Path to a directory designated by `input_dir`.
  - `output_dir`: Path to a directory designated by `output_dir`.
  - `divide_rate`: Used for a rate of data dividing designated by `divide_rate`.
  - `random_state`: Used for a seed value for sampling dividing designated by `random_state`.


##### load_test_data
  This function loads a data for testing. The outputs are images and labels for testing. You can omit implementing this function if you follow the file format standard. Parameters are as below.
  - `data_dir`: Path to a directory designated by `data_dir`.
  
##### load_repair_data
  This function loads a data for repairing. The outputs are images and labels for repairing. You can omit implementing this function if you follow the file format standard. Parameters are as below.
  - `data_dir`: Path to a directory designated by `data_dir`.

##### set_extra_config
  If you want to add any extra configs, you can implement here. For example, if you add a new option `new_option`, you can get the value of `new_option` from `kwargs` as below.  This function is called after generating this object in `cli.py`.

```python
  def set_extra_config(self, **kwargs):
      if 'new_option' in kwargs:
          self.new_option = kwargs['new_option']

  ```

##### (optional) Define `__all__`
You should define `__all__` at `__init__.py` if you develop your dataset as package to make it avaliable via cli.
You can define `__all__` like example below:
```python
# at your_dataset_pacakge/__init__.py
from .your_files import YourDataset, YourDataset2, ...

__all__ = ["YourDataset", "YourDataset2", ...]

```

See official docs to reference about `__all__` (https://docs.python.org/3/tutorial/modules.html?highlight=import#packages)

#### Implementing a new util functions

If you want to add new util functions like drawing a data chart or dividing a dataset, follow the steps below:

- Create new file under `src/repair/utils`. You can call this util funtion by designate the filename without suffix (.py).
- Define `run()` function. This function is the entrypoint of this utils. `run()` should accept `**kwargs` to pass some arguments such like `output_dir`.
- Then you can call this util function by running commands as below:

```bash
$ repair utils <util filename> --option1 value1 --option2 value2 ...
```

After implementing a new function, you should update the [user manual](https://github.com/jst-qaml/eAI-Repair/tree/master/docs/user_manual#utilities).

#### Modifying Documents

After enabling the model, you should updating a [user manual](https://github.com/jst-qaml/eAI-Repair/tree/master/docs/user_manual#dataset). Adding a brief description for the added dataset and how to get and place the image files. In addition, you should add the name of created dataset to `dataset` in the option table (including any other sections like train, test, target, repair, and so on). Also add a new option if you created.

If you modify the base class `dataset.py` and add/remove functions, you should modify this document besides the user manual.

### Adding a new model

#### Implementing a new model

When you want to add new model, follow the steps below.

##### Make module or packge
Place your model module or package under `src/repair/model`.
If you make new model as package, you should place `__init__.py` under `src/repair/model/<your model>`.
File structure should be like below after you place file(s).

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

##### Implement model class
Define your model class and inherit `RepairModel` base class.

```python
from repair.core.model import RepairModel 


class YourModel(RepairModel):
    ...

```

Implement methods below:

##### get_name
This function returns name that is used to load your model from repair cli.
Default value is class name of your model.
You can override this function to return the name you want to provide.

Example:
```python
class YourModel(RepairModel):
    @classmethod
    def get_name(cls) -> str:
        # default value is `YourModel`
        return "my_model"

```
In this example, you can load this model in cli like `repair train --model=my_model ...`.

##### compile
  This function is called for compiling a model in `train` command. The model is thought to be created by using keras [sequential model](https://www.tensorflow.org/guide/keras/sequential_model) or [function API](https://www.tensorflow.org/guide/keras/functional). Parameters are as below.
  - `input_shape`: Shape of model inputs, that is, the size of input images for training
  - `output_shape`: Shape of model outputs, that is, the number of classes of a dataset for training.

##### set_extra_config
  If you want to add any extra configs, you can implement here. For example, if you add a new option `new_option`, the value of it can be gotten from `kwargs` as below.  This function is called after generating this object in `cli.py`.

```python
  def set_extra_config(self, **kwargs):
      if 'new_option' in kwargs:
          self.new_option = kwargs['new_option']

  ```

##### (optional) Define `__all__`
You should define `__all__` at `__init__.py` if you develop your model as package to make it avaliable via cli.
You can define `__all__` like example below:
```python
# at your_model_pacakge/__init__.py
from .your_files import YourModel, YourModel2

__all__ = ["YourModel", "YourModel2", ...]

```

See official docs to reference about `__all__` (https://docs.python.org/3/tutorial/modules.html?highlight=import#packages)

#### Modifying Documents

After enabling the model, you should update a [user manual](https://github.com/jst-qaml/eAI-Repair/tree/master/docs/user_manual#3-dnn-model). Adding the name of created model to `model` in the option table. Also add a new option if you created.

If you modify the base class `model.py` and add/remove functions, you should modify this document besides the user manual.

### Adding a new method

If you want to implement new method, there are two options.

#### Creating a new PR to intend to be merged in to main stream 

1. Clone this repository
2. Implement the new method
3. Add test for the new method
4. Create a PR

See `Implementing a new method` section of `CONTRIBUTING_FOR_METHOD_IMPLEMENTERS.md` to learn how to implement a new method.

#### Implementing a new method and want to manage at out of this repository

See `CONTRIBUTING_FOR_METHOD_IMPLEMENTERS.md`.


### Code Formatting

This repository use [`black`](https://github.com/psf/black) to format code. We recommend setting up black to run on save. 

To run black mannualy, run `black .` at project root directory for all code, or run `black <path/to/file_or_dir>` for specific file(s).


### Linting

This repository use ['ruff'](https://github.com/charliermarsh/ruff) to lint code.

To run ruff mannualy, run `ruff check --fix .` at project root directory.

If you want to know what the error code means, run `ruff rule <Error codes>` to show description. For example, `ruff rule W291` shows description for trailing-whitespace(W291).


### Pre-Commit

We recommend you install the [`pre-commit`](https://pre-commit.com/) hooks provided in this repository. Theses hooks automatically run commands defined in `.pre-commit-config.yaml` before each time you run `git commit`. 

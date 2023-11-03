# Contributing Guidelines for Util Function Implementers

When you want to add new util functions, follow the steps below.

## Implementing Util Functions

If you want to add new util functions like drawing a data chart or dividing a dataset, follow the steps below:

### Creating Modules

- Create a new file under `src/repair/utils`. This toolkit loads your util funtion by using the filename without suffix (`.py`).
- Define a `run()` function as an entrypoint of your util function. The `run()` should accept `**kwargs` to pass arguments such like `output_dir`.
- Implement the code to parse the arguments. For example below, `option1` and `option2` are required and optional, respectively.

Example: `src/repair/utils/your_util_func.py`
```python
def run(**kwargs):
    """Your util functions."""
    if "option1" in kwargs:
        option1 = Path(kwargs["option1"])  # e.g., output_dir
    else:
        raise TypeError("Require option1")
    option2 = kwargs["option2"] if "option2" in kwargs else 0

```

Then users can call your util function by running commands as below:

```shell-session
(venv-name) $ repair utils your_util_func --option1 value1 --option2 value2 ...
```

## Updating Documents

You need to update the [Section Utilities](../docs/user_manual#7-utilities) in the user manual.
Create a subsection and add a brief description about your util function and how to use it.
This subsection is expected to contain precondition, description of options, example usage and results, and so on.

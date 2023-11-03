# Contributing to eAI-Repair-Toolkit

The following is a set of guidelines for those who contribute to eAI-Repair-Toolkit development.
These are mostly guidelines, not rules. Use your best practices, and feel free to propose changes to this document in pull requests.


#### Table Of Contents

[How Can I Contribute?](#how-can-i-contribute)
  * [Creating Issues](#creating-issues)
  * [Creating Branches](#creating-branches)
  * [Creating Pull Requests](#creating-pull-requests)

[How Can I Change the Code?](#how-can-i-change-the-code)
  * [Python Runtime](#python-runtime)
  * [Package Manager](#package-manager)
  * [Developer Tools](#developer-tools)
  * [Code Formatting](#code-formatting)
  * [Code Linting](#code-linting)
  * [Testing](#testing)
  * [License](#license)


## How Can I Contribute?

This repository applies issue centric development.
Before fixing a bug, adding a new feature or any other changing to this repository, you should create a related issue at first.
Then you create a branch related to the issue and create a pull request with the branch to close the issue.

### Creating Issues

- Reporting Bugs

  If you find a bug, please report it with the template as below.
  You should write `Overview`, `How to Reproduce` and `Expected Behavior` to make developers understand what happens precisely.
  The description of the way of reproducing the bug would be better as minimal as possible.
  Attach helpful `Screenshots` if you have.
  Please write any other information, like your idea for causing the bug or fixing it, in `Miscellaneous`.

  <details><summary>Show template </summary>

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

- Requesting New Features

  You should write `Request` at least to let developers know what you want.
  It would be better to write `as is` and `to be`, or background of the request in addition to the request itself.
  Write your implementing idea for the request in `How to Implement` or `Alternatives` if you have. 

  <details><summary>Show template</summary>

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

- Asking Questions

  If you have a question about this repository, ask it with the template as below. 

  <details><summary>Show template</summary>

  ```
  ## Question

  Describe insightful troubles you have.

  ## Miscellaneous

  ```

  </details>

- Maintaining Project and Repository 

  If you find some typos, request to manage packages used in this project, or request to configure some developer tools, ask it with the template as below.

  <details><summary>Show template</summary>

  ```
  ## Description

  Describe typos you found, what you request to add/update/remove packages or what you want to configure tools here.


  ## Miscellaneous

  ```

  </details>

- Refactoring the Code

  If you want to change code to improve performance, maintenability, etc., ask it with the template as below.

  <details><summary>Show template</summary>

  ```
  ## Description

  Describe the purpose of refactoring you will do, like improving performance, maintenability etc.
  You can also describe more detailed extra information to AsIs and ToBe section.

  ## As Is

  ## To Be

  ```

  </details>

Also you can create other types of issues without using the templates above.

### Creating Branches

Before creating a branch, you should create an issue related to it.
Developers want to know what is proceeding in this repository.
Therefore, the name of branch should contains the issue number and the class of changes like `feat`, `fix`, `docs`, `refactor`, `chore`, and `tests`. 

- `feat` : Creating new features and improvement.
- `fix` : Fixing bugs.
- `docs` : Modifying or creating documents.
- `refactor` : Refactoring the code.
- `chore` : Other changes like re-configuring virtual environments.
- `tests` : Create or add tests.

For example, the branch named `feat-70` is a branch that adding a new feature about an issue #70.
You can also attach more concrete contents after the branch name  (like `feat-70-add-hydranet-model`) .

### Creating Pull Requests

Before creating a pull request, you should create an issue related to it.
Reviewers want to know why this pull request has created, and these kinds of information should be stored in issues.
Pull requests should be created based on the template as below.

#### Pull Request Titles

Then, you should modify the PR title to keep semantic.
You should denote the type of PR at the head of title.

Template:

```
type(scope): title
```

| Item | Description | Remark |
| :--: | :---------- | :-----: |
| `type` | Name of PR type. Acceptable types are the same as those listed in [Creating a Branch](#creating-a-branch). | Required |
| `scope` | You can also denote the scope of impact of your PR. Name of scope should be the name of modules or directories. | Optional |
| `title` | Main title of your PR | Required |

Examples:

```
# without scope
feat: add new repair model
docs: fix typo in user manual

# with scope
fix(method): add missing options for arachne
tests(dataset): add tests for gtsrb
```

##### Available Scopes

You can use the terms below as PR scopes.
If you have trouble choosing a scope, you can omit it.
Or consider restructuring your PR so that you can choose the scope.
You are also welcome to suggest to add extra scopes.

- dataset
- docs
- method
- model
- tests

#### Pull Request Contents

You should write an overview of what you do and the issue number to be closed in `Details`.
Before creating pull requests, you have to do tests!
Write what you tested with respect to your changes in `Tests`.
Attaching screenshots or short gif files would be helpful to make reviewers understand the pull request.

At last, you should check `Check List` to ensure the quality of this repository.
Of course, `pytest`, `ruff` and `black` should be run and all the warnings should be corrected.

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

You may want to merge your own repair methods, models, and datasets in **main stream of this repository**.
The following guidelines help those who implement the core modules in addition to util functions.

- [Methods](METHOD.md)
- [Models](MODEL.md)
- [Datasets](DATASET.md)
- [Util functions](UTIL_FUNCS.md)

Additionally, it may require to expand current repair workflow, APIs, dependencies of virtual environments, and so on.
For such purposes as well, you can change the code in this repository.
Of course, fixing bugs and typos, code refactoring, security updates, etc. are very welcome.
Follow our coding standards below.

> [!NOTE]
> You can manage your own repair methods, models, and datasets in third-party repositories.
> Benefits of such outside repositories are, for example,
> making them private, adding other lisences, applying individual coding practices, and so on.
> For example, see [the Athena method](https://github.com/udzuki/eAI-RTK-Athena).
> In that case, you need to maintain those repositories as following latest eAI-Repair-Toolkit.
> It should be noted that we do not take on any responsibility of usage of them.

### Python Runtime

You must support Python versions compatible with this project. See [user manual](/docs/user_manual).

### Package Manager

You may use any Python package managers such as pip, [poetry](https://python-poetry.org/), [pdm](https://pdm.fming.dev/latest/) and so on.

> [!NOTE]
> We do NOT recommend to use [Conda](https://docs.conda.io/projects/conda/en/stable/) with pip
> because problems may occur due to difference between Conda and pip in ways to manage dependencies.
> If you want to use Conda, use it at your own risks.

### Developer Tools

We describe optional packages in [`pyproject.toml`](/pyproject.toml) for those who change and test the code in this repository.
Run the commands below to install them in your virtual environment.

```shell-session
(venv-name) $ pip install -e .[dev]
(venv-name) $ pip install -e .[test]
```

We recommend you to install [pre-commit](https://pre-commit.com/) that creates hooks in this repository.
These hooks work to automatically run commands defined in [`.pre-commit-config.yaml`](/.pre-commit-config.yaml) before each time you run `git commit`. 

```shell-session
(venv-name) $ pip install pre-commit
(venv-name) $ pre-commit install
```

### Code Formatting

This repository uses [Black](https://github.com/psf/black) to format the code.
We recommend setting up Black to run on save.
If you want to call Black manually, run the commands below:

```shell-session
(venv-name) $ black .                           # all
(venv-name) $ black src/repair/methods/arachne  # specific
```

### Code Linting

This repository uses [Ruff](https://github.com/charliermarsh/ruff) to lint the code.
If you want to call Ruff manually, run the commands below:

```shell-session
(venv-name) $ ruff check --fix .                          # all
(venv-name) $ ruff check --fix src/repair/methods/arachne # specific 
```

If you want to know what an error code means, run `ruff rule <error code>` to show its description.
For example, `ruff rule W291` shows description for ''trailing-whitespace (W291)''.

### Testing

We basically do not accept PRs that contains code changes but no tests.
A simple guideline for implementing test codes is as below:

- Test modules must to be under the `tests` directory.
- The filename of test modules must start with `test_`.
- The name of test classes must end with `Test`.

Reviewers will check not only test coverage measured in continuous integration but also implementation of test codes.
If your changes have reasonable excuses for degradation of test coverage, defend them through conversation with reviewers in PRs.
To test the code, run the commands below:

```shell-session
(venv-name) $ pytest --cov=src/             # all
(venv-name) $ pytest \
    tests/repair/methods/test_arachne.py \
    --cov=src/repair/methods/arachne        # specific
```

### License

You must agree that your implementation is published under [the license of eAI-Repair-Toolkit](../LICENSE).
If cannot, you manage it in third-party repositories on your own responsibility.

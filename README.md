# eAI-Repair-Toolkit

[![codecov](https://codecov.io/gh/jst-qaml/eAI-Repair-Toolkit/graph/badge.svg?token=L0QSZ9COAP)](https://codecov.io/gh/jst-qaml/eAI-Repair-Toolkit)

A toolkit for automated DNN repair. 

## Getting Started

### Prerequisites

* python >= 3.8
* pip >= 21.3

### Installing

```shell-session
$ python -m venv <path/to/venv>
$ source <path/to/venv>/bin/activate
(venv-name) $ pip install --upgrade pip
(venv-name) $ pip install -e .
```

### Usage

> [!NOTE]
> See [tutorial](docs/tutorial/) for an example usage with concrete DNN model and dataset.
> See [user manual](docs/user_manual) for details.

#### Issues Standing

The following items should be given in typical machine learning contexts.

* `model` : a DNN model (to be repaired)
* `dataset_test` : a dataset for testing `model` (and those repaired)

Let `dataset_test_A` be a subset of `dataset_test` containing data labelled ''A''.

```shell-session
(venv-name) $ repair test \
    --model_dir=${model} \
    --data_dir=${dataset_test_A}
...
accuracy: XX.XX%
```

If this `XX.XX%` does not satisfy your requirements for `model` (e.g., `YY.YY%` or larger), this toolkit can be used for repairing it.

#### Solution

This toolkit additionally requires:

* `dataset_repair` : an additional dataset for repairing `model`.
  This contents should be different from those in `dataset_test` (and a training dataset as well) to prevent overfitting.

This toolkit has a functionality to generate subsets of `dataset_repair` that `model` succeeds and fails to predict on each label.

```shell-session
(venv-name) $ repair target \
    --model_dir=${model} \
    --data_dir=${dataset_repair}
```

Let `dataset_repair_A_neg` be the subset that `model` fails to predict on the label ''A''.
In addition, let `dataset_repair_pos` be the subset that `model` succeeds to predict on all labels (called as a ''positive subset'').

Here you need to select a repair method to be applied.

* `method` : a repair method implemented in this toolkit. See [the list of repair methods](docs/user_manual#5-automated-repair)

Run this toolkit to localize suspicious neural weights in `model` which may cause misprediction on the label ''A''.

```shell-session
(venv-name) $ repair localize \
    --method=${method} \
    --model_dir=${model} \
    --target_data_dir=${dataset_A_neg}
```

Then, run it again to optimize the localized weights while preventing degradation in `model` by using the positive subset.

```shell-session
(venv-name) $ repair optimize \
    --method=${method} \
    --model_dir=${model} \
    --target_data_dir=${dataset_repair_A_neg} \
    --positive_inputs_dir={dataset_repair_pos}
```

Finally, this toolkit outputs a DNN model `model_repaired` that is a repaired candidate of `model`.
Let's check it.

```shell-session
(venv-name) $ repair test \
    --model_dir=${model_repaired} \
    --data_dir=${dataset_test_A}
...
accuracy: ZZ.ZZ%
```

If this `ZZ.ZZ%` is greater than the `YY.YY%`,
you can obtain `model_repaired` that satisfies your requirements,
indicating this toolkit worked to repair `model`!

## Contributing

Please read [CONTRIBUTING](CONTRIBUTING) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see [the tags on this repository](https://github.com/jst-qaml/eAI-Repair-Toolkit/tags). 

## Authors

* [Yuta Maezawa](https://github.com/mzw) (Udzuki/NII)
* [Fuyuki Ishikawa](https://github.com/f-ishikawa) (NII)
* [Nobukazu Yoshioka](https://github.com/kaz3) (Waseda/NII)

See also the list of [contributors](https://github.com/jst-qaml/eAI-Repair-Toolkit/graphs/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

We quote [the license](https://doc.bdd100k.com/license.html#license) below because some tests and tutorial involve the data and labels of [BDD100K](https://bdd-data.berkeley.edu/).

<blockquote>
<p>Copyright ©2018. The Regents of the University of California (Regents). All Rights Reserved.</p>

<p>THIS SOFTWARE AND/OR DATA WAS DEPOSITED IN THE BAIR OPEN RESEARCH COMMONS REPOSITORY ON 1/1/2021</p>

<p>Permission to use, copy, modify, and distribute this software and its documentation for educational, research, and not-for-profit purposes, without fee and without a signed licensing agreement; and permission to use, copy, modify and distribute this software for commercial purposes (such rights not subject to transfer) to BDD and BAIR Commons members and their affiliates, is hereby granted, provided that the above copyright notice, this paragraph and the following two paragraphs appear in all copies, modifications, and distributions. Contact The Office of Technology Licensing, UC Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-7201, otl@berkeley.edu, http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.</p>

<p>IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.</p>

<p>REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED “AS IS”. REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.</p>
</blockquote>

## Acknowledgments

* This work was supported by [JST-Mirai Program](https://www.jst.go.jp/mirai/en/program/super-smart/index.html) Grant Number JPMJMI18BB, Japan.

----
&copy; 2023 [eAI Project](https://engineerable.ai/)

# Tutorial

We write this tutorial to facilitate your understanding of using eAI-Repair-Toolkit for repairing DNN models.

## Setup Toolkit

First of all, you need to create a running environment for eAI-Repair-Toolkit. See [user manual](/docs/user_manual#virtual-environment). We give concrete commands below.

> [!NOTE]
> Suppose that your current directory is that where you cloned eAI-Repair-Toolkit.

```shell-session
$ python -m venv .venv
$ source .venv/bin/activate
(.venv)$ pip install --upgrade pip
(.venv)$ pip install -e .
```

## Prepare Model and Dataset

We selected VGG-16[^1] and Berkeley Deep Drive (BDD)[^2] as example model and dataset in terms of feasibility on contexts of machine learning based autonomous driving systems.

### Model

For simplicity, we provide a concrete VGG-16 model trained by using the BDD dataset. Use this model as a target DNN model to repair in this tutorial. This model is saved by spliting into `docs/tutorial/models`. To unzip the model, run the commands below:

```shell-session
$ cd docs/tutorial/models
$ cat splited.* > Merged.zip
$ unzip Merged.zip
```

You can find this model at `docs/tutorial/models/model`.

### Dataset

#### Download Dataset

Go to [Berkeley Deep Drive datasets portal](https://bdd-data.berkeley.edu) and download by pushing `100K Images` and `Detection 2020 Labels` buttons. Then, unzip the downloaded files to a directory (e.g., `eAI-Repair-Toolkit/datasets/bdd100k`).

#### Prepare Training and Test Dataset

We preprocess the BDD dataset to fit the following purposes in this tutorial.

1. We focus on an image classification task. Since the dataset consists of images containing multiple objects, we extract each object into one labeled image.
2. We design eAI-Repair-Toolkit to repair DNN models, i.e., we need a **repair** dataset, in addition to train/test/val datasets to construct DNN models. Therefore, we use the BDD _val_ dataset as train/test/val datasets for model construction. As for a **repair** dataset, we use a part of the BDD _train_ dataset. Note that we do not use the BDD _test_ dataset because it does not have any label data.

We extract images containing one object with its label from the BDD _val_ dataset. The following commands work for it.

```shell-session
(.venv)$ repair create_category_dir \
    --input_dir=datasets/bdd100k/images/val \
    --label_file=datasets/bdd100k/labels/det_20/det_val.json \
    --output_dir=docs/tutorial/categories/val
```

We exclude the 8 labels because the numbers of images with these labels are 1,000 or less. This exclusion may be reasonable on the machine learning context because it tends to require the large number of data.

```shell-session
(.venv)$ repair create_image_dir \
    --category_dir=docs/tutorial/categories/val \
    --output_dir=docs/tutorial/data/images/val \
    --num=70000 \
    --excluded_labels='other vehicle,other person,trailer,train,bicycle,motorcycle,bus,rider'
|               label|   num|
|--------------------|------|
|                 car| 38696|
|               truck|  1597|
|          pedestrian|  5052|
|        traffic sign| 13066|
|       traffic light| 10116|
|               total| 68527|
```

```shell-session
(.venv)$ repair prepare bdd-objects \
    --input_dir=docs/tutorial/data/images/val \
    --output_dir=docs/tutorial/data \
    --single_dir=True \
    --data_ratio=0.5,0.2,0.3,0 \
    --random_state=3
100%|██████████████████████████████████████████████████████████████████████████| 34263/34263 [05:39<00:00, 100.88it/s]
train_images: (34263, 32, 32, 3)
train_labels: (34263, 13)
train_image_ids: (34263,)
train_attribute_ids: (34263,)
100%|███████████████████████████████████████████████████████████████████████████| 13705/13705 [02:22<00:00, 96.31it/s]
val_images: (13705, 32, 32, 3)
val_labels: (13705, 13)
val_image_ids: (13705,)
val_attribute_ids: (13705,)
100%|███████████████████████████████████████████████████████████████████████████| 20558/20558 [04:19<00:00, 79.09it/s]
test_images: (20558, 32, 32, 3)
test_labels: (20558, 13)
test_image_ids: (20558,)
test_attribute_ids: (20558,)

```

You can find `test.h5`, `train.h5` and `val.h5` in `docs/tutorial/data`. We used this `train.h5` to construct the example model described in the Model Section. The `test.h5` is used for evaluating how eAI-Repair-Toolkit works to repair the example model in the following Sections.

#### Prepare Repair Dataset

As well as the above, we extract a dataset from the BDD _train_ dataset as a **repair** dataset that we use to repair the example model with eAI-Repair-Toolkit.

```shell-session
(.venv)$ repair create_category_dir \
    --input_dir=datasets/bdd100k/images/val \
    --label_file=datasets/bdd100k/labels/det20/det_train.json \
    --output_dir=docs/tutorial/categories/train
```

As described in the Repair Section later, we set repair requirements relevant to `pedestrian`, so we extract only its data to reduce execution time. Run the commands below:

```shell-session
(.venv)$ repair create_image_subset \
    --category_dir=docs/tutorial/categories/train \
    --output_dir=docs/tutorial/data/images/train \
    --num=13000000 \
    --excluded_labels='car,bus,bicycle,motorcycle,rider,trailer,train,truck,traffic light,traffic sign,other vehicle,other person'
|               label|   num|
|--------------------|------|
|          pedestrian| 92159|
|               total| 92159|
```

```shell-session
(.venv)$ repair prepare bdd-objects \
    --input_dir=docs/tutorial/data/images/train \
    --output_dir=docs/tutorial/data \
    --single_dir=True \
    --data_ratio=0,0,0,1
100%|██████████████████████████████████████████████████████████████████████████| 92159/92159 [11:42<00:00, 131.20it/s]
repair_images: (92159, 32, 32, 3)
repair_labels: (92159, 13)
repair_image_ids: (92159,)
repair_attribute_ids: (92159,)
```

You can find `repair.h5` in `docs/tutorial/data`. However, `repair.h5` is too big to use in a single repair. To make it smaller, here is the sample Python script to split a dataset into 100 sized dataset. We created 10 chuncked datasets for running 10 repair iterations in this tutorial.

```python
"""File: split_repair_dataset.py.

This is a sample script to split repair dataset into 10 of chunked dataset.
You can create a new script or modify this script to meet your requirements.

Examples
--------
> python scripts/split_repair_dataset repair.h5

> tree .
.
│
...
├── repair.h5
├── repairsets
│   ├── 0
│   │   └── repair.h5
│   ├── 1
│   │   └── repair.h5
│   ├── 2
│   │   └── repair.h5
│   ├── 3
│   │   └── repair.h5
│   ├── 4
│   │   └── repair.h5
│   ├── 5
│   │   └── repair.h5
│   ├── 6
│   │   └── repair.h5
│   ├── 7
│   │   └── repair.h5
│   ├── 8
│   │   └── repair.h5
│   └── 9
│       └── repair.h5
...
"""

import sys
from pathlib import Path

import h5py

DEFAULT_CHUNK_SIZE = 100


def run(file: str, chunk_size: int = DEFAULT_CHUNK_SIZE):
    repairset_root = Path(file).parent
    if not repairset_root.exists():
        raise ValueError("Invalid path")

    repairset_root /= "repairsets"
    repairset_root.mkdir(exist_ok=True)

    hf = h5py.File(file)
    dataset_len = len(hf["labels"])
    for i in range(10):
        if i * chunk_size > dataset_len:
            print("program finished with short data.")
            break

        chunk_root = repairset_root / str(i)
        chunk_root.mkdir()

        if (i + 1) * chunk_size > dataset_len:
            chunked_imgs = hf["images"][i * chunk_size :]
            chunked_lbls = hf["labels"][i * chunk_size :]
        else:
            chunked_imgs = hf["images"][i * chunk_size : (i + 1) * chunk_size]
            chunked_lbls = hf["labels"][i * chunk_size : (i + 1) * chunk_size]

        with h5py.File(chunk_root / "repair.h5", "w") as f:
            f.create_dataset("images", data=chunked_imgs)
            f.create_dataset("labels", data=chunked_lbls)


if __name__ == "__main__":
    file = sys.argv[1]
    run(file)
```

Now we are ready to repair the example model with eAI-Repair-Toolkit!  

## Repair

### Scenario

We set up the following these requirements in this tutorial:

1. Overall accuracy should be 85 % or greater
2. Accuracy for pedestrian should be 75 % or greater
3. Accuracy for car should be 90 % or greater

### Evaluate Model to Repair

At first, let's evaluate overall accuracy of the example model. 

```shell-session
(.venv)$ repair test --model_dir=docs/tutorial/model --data_dir=docs/tutorial/data
accuracy: 89.67%
```

To get more detailed results, run the following command.

```shell-session
(.venv)$ repair utils draw_radar_chart \
    --model_dir=docs/tutorial/model \
    --data_dir=docs/tutorial/data \
    --output_dir=docs/tutorial/data
```

You can find `results.json` at `docs/tutorial/data`. The `results.json` consists of an array of tuple of a label and its numbers of images that the example model succeeded to classify, failed to classify, and score (accuracy) of classification.

> [!NOTE]
> Labels 2 and 6 are for car and pedestrian, respectively.

```json
[
    [
        "12",
        {
            "success": 97,
            "failure": 389,
            "score": 19.958847736625515
        }
    ],
    [
        "2",
        {
            "success": 11140,
            "failure": 471,
            "score": 95.94350185169236
        }
    ],
    [
        "6",
        {
            "success": 1132,
            "failure": 398,
            "score": 73.98692810457516
        }
    ],
    [
        "8",
        {
            "success": 2502,
            "failure": 439,
            "score": 85.07310438626318
        }
    ],
    [
        "9",
        {
            "success": 3563,
            "failure": 427,
            "score": 89.29824561403508
        }
    ]
]
```

From these results, the example model meets requirements 1 and 3, but not 2. Therefore we apply a repair method in eAI-Repair-Toolkit to improve the accuracy for pedestrian.

### Repair Model

We update the requirements of repair.

1. Degradation of overall accuracy should be 1 % or lesser
1. Accuracy for pedestrain should be 75 % or greater
3. Accuracy for car should keep 90 % or greater

#### Prepare Repair Dataset

At first, we divide the repair dataset into negative and positive datasets that the example succeeded and failed to classify, respectively. Run the command below.

> [!NOTE]
> The `.../repairsets/0` is the first chunk of the repair dataset. It contains all kinds of labeled images.

```shell-session
(.venv)$ repair target --model_dir=docs/tutorial/model --target_data=docs/tutorial/data/repairsets/0
```

You can find `negative` and `positive` directories under `.../repairsets/0`. The `negative` directory contains the images that the example model failed to classify and they are divided by each label. For example, `.../repairsets/0/negative/6` contains the `pedestrian` images that the example model failed to classify in the first chunk of the repair dataset.

#### Apply Repair Method

Then we apply the Arachne's localization functionality to localize weights that may cause the misclassification on the `pedestrian` images.

```shell-session
(.venv)$ repair localize \
    --method=Arachne
    --model_dir=docs/tutorial/model 
    --target_data_dir=docs/tutorial/data/repairsets/0/negative/6
```

After that, we also apply the Arachne's optimization functionality to search modified weights that improve accuracy for pedestrian according with prevent degradation on car and overall accuracy.

```shell-session
(.venv)$ repair optimize \
    --method=Arachne
    --model_dir=docs/tutorial/model
    --target_data_dir=docs/tutorial/data/repairsets/0/negative/6
    --positive_inputs_dir=docs/tutorial/data/repairsets/0/positive
    --num_iterations=3 \
    --num_particles=3
```

Finally, we can obtain repaired model at `docs/tutorial/data/repairsets/0/negative/6/repair`.

#### Evaluate Repaired Model

Let's evaluate the repaired model with `outputs/tutorial/test.h5`, i.e., the same dataset used in evaluation of the example model to repair.

```shell-session
(.venv)$ repair utils draw_radar_chart \
    --model_dir=docs/tutorial/data/repairsets/0/negative/6/repair \
    --input_dir=outputs/tutorial \
    --output_dir=docs/tutorial/data/repairsets/0 \
    --target_data=test.h5
```

Then let's check the results. 

```json
[
    [
        "12",
        {
            "success": 102,
            "failure": 384,
            "score": 20.987654320987655
        }
    ],
    [
        "2",
        {
            "success": 11097,
            "failure": 514,
            "score": 95.57316337955388
        }
    ],
    [
        "6",
        {
            "success": 1149,
            "failure": 381,
            "score": 75.09803921568627
        }
    ],
    [
        "8",
        {
            "success": 2507,
            "failure": 434,
            "score": 85.24311458687521
        }
    ],
    [
        "9",
        {
            "success": 3569,
            "failure": 421,
            "score": 89.4486215538847
        }
    ]
]
```

Yay! The repaired model satisfies all of our requirements! Although the accuracy for car has decreased slightly, it would be acceptable. In general, it is interesting on a tradeoff where repairing for certain labels may degrade accuracy for other labeled datasets.

#### Iterate Repair Cycle

Let's see if further repairs will improve the example model. Here is the sample bash script below to run repair iterations. We assume these iterations as new data collection for repair.

> [!NOTE]
> This sample script performs all repair iterations including the initial repair attempt.

```bash
#!/bin/bash

source .venv/bin/activate

data_dir=docs/tutorial/data
repair_dir=$data_dir/repairsets/

for i in {0..9} ; do
    repairset_dir=$repair_dir/$i/
    if test $i -eq 0 ; then
        target_model_dir=docs/tutorial/model
    else
        target_model_dir=$repair_dir/$((i-1))/negative/6/repair
    fi

    repaired_model_dir=$repair_dir/$i/negative/6/repair
    negative_data_dir=$repairset_dir/negative/6
    positive_data_dir=$repairset_dir/positive

    repair target \
        --model_dir=$target_model_dir \
        --data_dir=$repairset_dir

    repair localize \
        --method=Arachne \
        --model_dir=$model_dir \
        --target_data_dir=$negative_data_dir \

    repair optimize \
        --method=Arachne \
        --model_dir=$model_dir \
        --target_data_dir=$negative_data_dir \
        --positive_inputs_dir=$positive_data_dir \
        --num_iterations=3 \
        --num_particles=3 \

    repair utils draw_radar_chart \
        --model_dir=$repaired_model_dir \
        --input_dir=$data_dir \
        --target_data=test.h5 \
        --output_dir=$repairset_dir \

    echo
done
```

We show the results of repair iterations at Table 1. From these results, the repaired model at the interation 5 is the best in terms of the accuracy for pedestrian. It may indicate limitation of repair performance. Properly using retraining and repairing DNN models may be promising to satisfy various requirements on machine learning systems.

<table>
    <thead>
    <tr>
        <th>iteration number</th>
        <th>correct</th>
        <th>incorrect</th>
        <th>accuracy</th>
    </tr>
    </thead>
    <tbody>
        <tr>
            <td>0 (base)</td>
            <td>1132</td>
            <td>398</td>
            <td>73.99</td>
        </tr>
        <tr>
            <td>1</td>
            <td>1149</td>
            <td>381</td>
            <td>75.10</td>
        </tr>
        <tr>
            <td>2</td>
            <td>1143</td>
            <td>387</td>
            <td>74.71</td>
        </tr>
        <tr>
            <td>3</td>
            <td>1167</td>
            <td>363</td>
            <td>76.27</td>
        </tr>
        <tr>
            <td>4</td>
            <td>1158</td>
            <td>372</td>
            <td>75.69</td>
        </tr>
        <tr>
            <td>5</td>
            <td>1173</td>
            <td>357</td>
            <td>76.67</td>
        </tr>
        <tr>
            <td>6</td>
            <td>1158</td>
            <td>372</td>
            <td>75.69</td>
        </tr>
        <tr>
            <td>7</td>
            <td>1161</td>
            <td>369</td>
            <td>75.88</td>
        </tr>
        <tr>
            <td>8</td>
            <td>1160</td>
            <td>370</td>
            <td>75.82</td>
        </tr>
        <tr>
            <td>9</td>
            <td>1168</td>
            <td>362</td>
            <td>76.34</td>
        </tr>
        <tr>
            <td>10</td>
            <td>1166</td>
            <td>364</td>
            <td>76.21</td>
        </tr>
    </tbody>
    <caption>Table 1. The accuracy for pedestrian of each repair iterations</caption>
</table>

## Conclusion

In this tutorial, we showed how to use eAI-Repair-Toolkit to repair DNN models. From the results, it would be promising that a gradual repair of DNN models with a small amount of data can yield a model that meets requirements.


### References

[^1]: Very Deep Convolutional Networks for Large-Scale Image Recognition (ICLR 2015)
[^2]: BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning (CVPR 2020)

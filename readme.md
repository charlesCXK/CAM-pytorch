#### I. Introduction

Use pytorch 0.4.0 and python3 to implemennt CAM (class avtivation mapping). Just a simple lab, acting on cifar-10 dataset.

<a href='https://blog.csdn.net/weixin_40955254/article/details/81191896'>Click here for more</a>

<br>

#### II. Usage

```shell
$ python3 cam.py
```

And the result will be saved in directory **res**.

<br>

#### III. Description for packages and files.

— res/ — 

```
 A directory where the results are placed.
```

— cifar-10-batches-py/ — 

```
Cifar-10 data set
```

— cam.py — 

```
Run and save the result of CAM method.
```

— train.py  — 

```
Train the net I defined with 20 epoches and SGD.
```

— params.pkl — 

```
Model-state saved by pytorch.
```

<br>

IV. Result

Some of the results are shown below. White regions are what we interested in.

<img src='res/4.png'>

<img src='res/13.png'>

<br>

#### V. To do

Use larger dataset, for now, the size of feature map before GAP layer is only 8*8, and I resize it to 224\*224. There must be much loss of information during this process.
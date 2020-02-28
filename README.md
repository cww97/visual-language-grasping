# Visual Language Grasping

蔡队牛逼

## 文件说明

```
|- docs/                         # 自己的一些笔记和文档
|- downloads/                    # 下载原作者预训练模型的脚本和模型
|- envs/                         # environemts(sim/real)
    |- objects/                  # sim环境中物体模型文件(`.obj`)
    |- real/                     # real环境中的测试脚本
        |- capture.py            # 测试真实环境中的相机能正确传输数据
        |- calibrate.py          # 校准real环境中的相机位置
        |- debug.py              # 在real环境下测试机械臂的连接与动作
        |- touch.py              # 测试外部相机，有一个UI
    |- realsense/            # 跑真机械臂的jdk环境(`.cpp`)
    |- simulation/           # vrepAPI和场景文件(`.ttt`)
    |- create.py                 # sim中创建自己的测试用例
    |- robot.py                  # sim和real的机械臂都在里面
|- logs/                         # log嘛你懂我意思吧
|- utils/                        # 常用工具/全局变量
    |- __init__.py               # 原utils.py
    |- config.yaml               #
    |- config.py                 #
    |- evaluate.py               # 评估成功率
    |- plot.py                   # 画图,loss, 成功率等等
|- main.py                       # 
|- trainer.py                    # 
|- models.py                     # 
```

##  Plan

### 数据生成

[八个obj](docs/csl_note.md)分为5类形状, 每个形状的obj表为

```python
[
    ['0.obj'],                                # 三棱柱
    ['1.obj'],                                # 拱形
    ['2.obj', '3.obj', '4.obj', '6.obj'],     # 长方体
    ['7.obj'],                                # 半圆柱
    ['8.obj'],                                # 圆柱
]

```

随机扔下k(k=2~5)个物体，保证每种颜色每类形状至多出现一次，从k个物体中随机选一个，获取其形状与颜色，生成句柄 `pick up the [COLOR] [SHAPE]`, `pick up`这个词也可以替换

生成`scene-instruction`一组数据

### 算法

原有的对图像的处理框架不变，加入一个`LSTMEncoder`处理instruction，然后用图像feature与Encoder的隐藏层作为Decoder的输入，Decoder的hidden_state与Encoder的hidden_state再attention，最后输出action



## 运行配置

删去了原来的臃肿到爆炸的parser，改为读配置文件，默认在`utils/config.yaml`中读配置，建议复制该文件一份然后`-f/--file YOUR_CONFIG_FILE`，可以放在downloads这个不怕乱整的目录里去



### 继续之前的训练

`--load_snapshot` and `--continue_logging`

```
is_sim --push_rewards: true 
experience_replay: true
explore_rate_decay: true
save_visualizations: true
load_snapshot: true
snapshot_file: 'logs/YOUR-SESSION-DIRECTORY-NAME-HERE/models/snapshot-backup.reinforcement.pth' 
continue_logging: true 
logging_directory: 'logs/YOUR-SESSION-DIRECTORY-NAME-HERE' 
```

### 使用预训练模型 

`--snapshot_file`

```
is_sim: true
obj_mesh_dir: 'envs/objects/blocks' 
num_obj: 10
push_rewards: true
experience_replay: true
explore_rate_decay: true
is_testing: true
test_preset_cases: true 
test_preset_file: 'simulation/test-cases/test-10-obj-07.txt'
load_snapshot: true
snapshot_file: 'YOUR-SNAPSHOT-FILE-HERE'
save_visualizations: true
```



## Installation

This implementation requires the following dependencies (tested on Ubuntu 16.04.4 LTS): 

* Python 3 
* [PyTorch](http://pytorch.org/), [NumPy](http://www.numpy.org/), [SciPy](https://www.scipy.org/scipylib/index.html), [OpenCV-Python](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_tutorials.html). You can quickly install/update these dependencies by running the following (replace `pip` with `pip3` for Python 3):
  ```shell
  pip install numpy scipy opencv-python torch torchvision
  ```
* [V-REP](http://www.coppeliarobotics.com/) (simulation environment)

###  GPU Acceleration

决心修复可以在没有GPU的情况下运行的bug，假设你们已经有GPU，且配好了 [CUDA](https://developer.nvidia.com/cuda-downloads) 和 [cuDNN](https://developer.nvidia.com/cudnn). This code has been tested with CUDA 8.0 and cuDNN 6.0 on a single NVIDIA Titan X (12GB). Running out-of-the-box with our pre-trained models using GPU acceleration requires 8GB of GPU memory. Running with GPU acceleration is **highly recommended**, otherwise each training iteration will take several minutes to run (as opposed to several seconds). 原作者用的Titan X, 我们用的1080Ti.


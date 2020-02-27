# Visual Language Grasping

蔡队牛逼

## 文件说明

```
|- docs/                    # 自己的一些笔记和文档
|- downloads/               # 下载原作者预训练模型的脚本和模型
|- images/                  # README中的image文件
|- logs/                    # log嘛你懂我意思吧
|- objects/                 # sim环境中物体模型文件(`.obj`)
|- real/                    # real环境中的测试脚本
   |- capture.py            # 测试真实环境中的相机能正确传输数据
   |- calibrate.py          # 校准real环境中的相机位置
   |- debug.py              # 在real环境下测试机械臂的连接与动作
   |- touch.py              # 测试外部相机，有一个UI
|- realsense/               # 跑真机械臂的jdk环境(`.cpp`)
|- simulation/              # vrepAPI和场景文件(`.ttt`)
|- utils/                   # 常用工具/全局变量
   |- __init__.py           # 原utils.py
   |- evaluate.py           # 评估成功率
   |- plot.py               # 画图,loss, 成功率等等
   |- create.py             # sim中创建自己的测试用例
|- main.py                  # 
|- trainer.py               # 
|- models.py                # 
|- robot.py                 # sim和real的机械臂都在里面

```



## 常用命令

### 参数列表

Various training options can be modified or toggled on/off with different flags (run `python main.py -h` to see all options):

```shell
usage: main.py [-h] [--is_sim] [--obj_mesh_dir OBJ_MESH_DIR]
               [--num_obj NUM_OBJ] [--tcp_host_ip TCP_HOST_IP]
               [--tcp_port TCP_PORT] [--rtc_host_ip RTC_HOST_IP]
               [--rtc_port RTC_PORT]
               [--heightmap_resolution HEIGHTMAP_RESOLUTION]
               [--random_seed RANDOM_SEED] [--method METHOD] [--push_rewards]
               [--future_reward_discount FUTURE_REWARD_DISCOUNT]
               [--experience_replay] [--heuristic_bootstrap]
               [--explore_rate_decay] [--grasp_only] [--is_testing]
               [--max_test_trials MAX_TEST_TRIALS] [--test_preset_cases]
               [--test_preset_file TEST_PRESET_FILE] [--load_snapshot]
               [--snapshot_file SNAPSHOT_FILE] [--continue_logging]
               [--logging_directory LOGGING_DIRECTORY] [--save_visualizations]
```

### Instructions



    ```shell
    python main.py --is_sim --obj_mesh_dir 'objects/blocks' --num_obj 10 \
        --push_rewards --experience_replay --explore_rate_decay \
        --is_testing --test_preset_cases --test_preset_file 'simulation/test-cases/test-10-obj-07.txt' \
        --load_snapshot --snapshot_file 'downloads/vpg-original-sim-pretrained-10-obj.pth' \
        --save_visualizations
    ```

### 继续之前的训练

`--load_snapshot` and `--continue_logging`

```shell
python main.py --is_sim --push_rewards --experience_replay --explore_rate_decay --save_visualizations \
    --load_snapshot --snapshot_file 'logs/YOUR-SESSION-DIRECTORY-NAME-HERE/models/snapshot-backup.reinforcement.pth' \
    --continue_logging --logging_directory 'logs/YOUR-SESSION-DIRECTORY-NAME-HERE' \
```

### 使用预训练模型 

`--snapshot_file`

```shell
python main.py --is_sim --obj_mesh_dir 'objects/blocks' --num_obj 10 \
    --push_rewards --experience_replay --explore_rate_decay \
    --is_testing --test_preset_cases --test_preset_file 'simulation/test-cases/test-10-obj-07.txt' \
    --load_snapshot --snapshot_file 'YOUR-SNAPSHOT-FILE-HERE' \
    --save_visualizations
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

## 文件说明 - old

前面三个文件感觉相当混乱,整理ing
- `main.py`: 
- `trainer.py`:
- `models.py`: 
- `utils.py`: 看名字就知道干啥的了吧
- `robot.py` sim和real的机械臂都在里面

### Tools

- `evaluate.py`: 评估成功率
- `plot.py`, 画图,loss, 成功率等等
- `create.py`: sim中创建自己的测试用例

### Real

- `real/capture.py`: 测试真实环境中的相机能正确传输数据
- `calibrate.py`: 校准real环境中的相机位置
- `debug.py`: 在real环境下测试机械臂的连接与动作
- `touch.py`: 测试外部相机，有一个UI点RGB-D图像上的一个点，然后机械臂移动过去

### 其他目录说明

- docs: 自己的一些笔记和文档
- downloads: 下载原作者预训练模型的脚本和模型
- images: README中的image文件
- logs: log嘛你懂我意思吧
- objects: sim环境中物体模型文件(`.obj`)
- real: real环境中的相机与测试脚本，感觉可以把其他的real的也扔进去
- realsense: 跑真机械臂的jdk环境(`.cpp`)
- simulation: vrepAPI和场景文件(`.ttt`)
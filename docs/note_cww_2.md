

codes are [here](https://github.com/peteanderson80/Matterport3DSimulator/tree/master/tasks/R2R)


## Directory Structure

- `env.py`: Wraps the simulator and adds language instructions, with several simplifications -- namely discretized heading / elevation and pre-cached image features. This is not intended to be a standard component, or to preclude the use of continous camera actions, end-to-end training etc. Use the simulator and the data as you see fit, but this can provide a starting point.
- `utils.py`: Text pre-processing, navigation graph loading etc.
- `eval.py`: Evaluation script.
- `model.py`: PyTorch seq2seq model with attention.
- `agent.py`: Various implementations of an agent.
- `train.py`: Training entrypoint, parameter settings etc.
- `plot.py`: Figures from the arXiv paper.



## review

`Seq2SeqAgent` in agent.py

内部格式转换的函数: `_sort_batch()`, `_feature_variable()`, `_teacher_action()`, 

baseAgent里面的rollout没有实现

逻辑是

train -> agent -> model

- model里面包含一个seq2seq的三个组件：LSTMEncoder, Attention, LSTMDecoder
- agent中的rollout函数为执行
- train里面主要做一些loss和acc的记录

对比之下，vpg的代码显的比较混乱，首先是屏幕宽度不够，一行五百多个字符谁顶得住

但是就算丑，也没有时间与精力全部去改他，分析一下需要改什么把

原来的代码的逻辑是：

main -> trainer -> models

main里的process_actions是执行模型决策的另一个进程，前面是大段的读取参数，`__main__`里面有，`main()`里面还有一遍，这些感觉都不该在main里面占大量的篇幅，感觉可以写进`utils`里或者额外开一个py文件

然后是`robot`, `trainer`, `logger`的定义，这些写在main里面怪合理的，上面的读参数不是说不合理，篇幅太多就该分出去了

再往后是`process_action()`建进程，与上面合并同类项

后面是一个`while True`的训练大循环，循环里：

- 先获取图片并save(20行)，
- 检查环境是否需要reset(20+行)
- `trainer.forward()`并送给另一个进程process(10行)
- 计算`no_change_count`(20行)
- 计算label，backward(13行)
- `experience_replay`(50行)
- `save logger`，save循环变量(32行)



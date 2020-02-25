

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

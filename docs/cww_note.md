





File | what
--- | ---
`robot.py` | control the robot, [here](robot.md)
`calibrate.py` | 校准
`create.py` | create a test case from a simulation environment
`debug.py` | debug in real world to make sure that the robot is available
`evaluate.py` | evaluate success rate
`main.py` | main training/testing loop & Parallel thread to process network output and execute actions
`models.py` | two networks (reactive/reinforcement)
`plot.py` | plot training sessions
`touch.py` | a toy ?
`trainer.py` |
`utils.py` |



## Recurrent Table 1

SIMULATION RESULTS ON RANDOM ARRANGEMENTS (MEAN %)

paper

Method |Completion |Grasp Success | Action Efﬁciency 
---|---|---|--
Grasping-only | 90.9 | 55.8 | 55.8 
P+G Reactive | 54.5 | 59.4 | 47.7 
VPG | 100.0 | 67.7 | 60.9

we did

Method | clearance |Grasp Success | Action Efﬁciency 
---|---|---|--
Grasping-only | 83.3 | 76.1 | 75 
P+G Reactive | 76.7 | 72.7 | 71.9  
VPG | 73.3 | 81.1 | 81.1


## test reactive models

```

python main.py --is_sim --obj_mesh_dir 'objects/blocks' --num_obj 30 \
    --push_rewards --experience_replay --explore_rate_decay \
    --is_testing  \
    --load_snapshot --snapshot_file 'logs/train-reinforcement-2019-07-04.18:02:52/models/snapshot-backup.reactive.pth' \
    --save_visualizations

python evaluate.py --session_directory 'logs/2019-08-27.12:22:52' --method 'reinforcement' --num_obj_complete 30

python main.py --is_sim --obj_mesh_dir 'objects/blocks' --num_obj 10 \
    --push_rewards --experience_replay --explore_rate_decay \
    --is_testing --test_preset_cases --test_preset_file 'simulation/test-cases/test-10-obj-07.txt' \
    --load_snapshot --snapshot_file 'downloads/vpg-original-sim-pretrained-10-obj.pth' \
    --save_visualizations
```

we have to add a '--method 'reactive' ', because its default is 'reinforcement'


heuristic bootstrap: interesting ??? what r u fucking doing




## Robot

`robot.py` is to control the robot(ur5) in simulation or real world
here is the list of their functions


```python
class Robot(object):

def __init__(self, 
	is_sim,                # true/false 
	obj_mesh_dir,          # 'objects/blocks', obj files
	num_obj,               # number of objects
	workspace_limits,      # diff between real and sim
	tcp_host_ip, 
	tcp_port, 
	rtc_host_ip, 
	rtc_port,
	is_testing,            # true/false
	test_preset_cases,     # 
	test_preset_file       # 
):
    Sim:
        Connect simulation
        Ugly codes about test_ojbs:{files, colors, positions, orientations}
        Read obj files and add them into memory(4 lists: obj_mesh_xxx)
        Then add objects into simulation
    Real:
        Tcp, rtc(ip:port), Close gripper, go home
    Init camera

# Sim
Def setup_sim_camera: 
def add_objects(self):
def restart_sim(self):
def check_sim(self): # by checking if gripper is within workspace
def get_task_score(self):     # ???
def check_goal_reached(self):
def get_obj_positions(self):
def get_obj_positions_and_orientations(self):
def reposition_objects(self, workspace_limits): #drop objs
	
# Both sim & real
def get_camera_data(self):
def close_gripper(self, async=False):
def open_gripper(self, async=False):
def grasp(self, position, heightmap_rotation_angle, workspace_limits):
def push(self, position, heightmap_rotation_angle, workspace_limits):
def move_to(self, tool_position, tool_orientation):

# real world
def get_state(self):
def guarded_move_to(self, tool_position, tool_orientation):
def move_joints(self, joint_configuration):
def go_home(self):
def check_grasp(self):
def restart_real(self):
def parse_tcp_state_data(self, state_data, subpackage):
	def parse_joint_data(data_bytes, byte_idx):
	def parse_cartesian_info(data_bytes, byte_idx):
	def parse_tool_data(data_bytes, byte_idx):
def parse_rtc_state_data(self, state_data):
```

## RL

in each step:
get any kind infomation for state, then action, at last backprop by reward.

in this main.py:

- get RGB-D image -> heightmap
- trainer.forward() -> execute action
- compute label -> backprop

what we are gonna do is modify the 'compute label' part:


import os
import numpy as np
import threading
import yaml


class Config(object):
    class ConfigLoader(object):
        __DEFAULT_CONFIG = os.path.join('utils', 'config.yaml')

        def __read_config_from_file(self, file):
            print('config file: ', file)
            if file is None:
                return
            with open(file) as f:
                for key, value in yaml.safe_load(f).items():
                    self.__dict__[key] = value

        def __init__(self, config_file):
            self.__read_config_from_file(self.__DEFAULT_CONFIG)
            if config_file:
                self.__read_config_from_file(config_file)

    _instance_lock = threading.Lock()

    # singleModule
    def __new__(cls, *args, **kwargs):
        if not hasattr(Config, '_instance'):
            with Config._instance_lock:
                if not hasattr(Config, '_instance'):
                    Config._instance = object.__new__(cls)
        return Config._instance

    def __init__(self, config_file):
        args = Config.ConfigLoader(config_file)
        # --------------- Setup options ---------------
        self.is_sim = args.is_sim # Run in simulation?
        self.obj_mesh_dir = os.path.abspath(args.obj_mesh_dir) if self.is_sim else None # Directory containing 3D mesh files (.obj) of objects to be added to simulation
        self.num_obj = args.num_obj if self.is_sim else None # Number of objects to add to simulation
        self.tcp_host_ip = args.tcp_host_ip if not self.is_sim else None # IP and port to robot arm as TCP client (UR5)
        self.tcp_port = args.tcp_port if not self.is_sim else None
        self.rtc_host_ip = args.rtc_host_ip if not self.is_sim else None # IP and port to robot arm as real-time client (UR5)
        self.rtc_port = args.rtc_port if not self.is_sim else None
        if self.is_sim:
            self.workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
        else:
            self.workspace_limits = np.asarray([[0.3, 0.748], [-0.224, 0.224], [-0.255, -0.1]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
        self.heightmap_resolution = args.heightmap_resolution # Meters per pixel of heightmap
        self.random_seed = args.random_seed

        # ------------- Algorithm options -------------
        self.method = args.method # 'reactive' (supervised learning) or 'reinforcement' (reinforcement learning ie Q-learning)
        self.push_rewards = args.push_rewards if self.method == 'reinforcement' else None  # Use immediate rewards (from change detection) for pushing?
        self.future_reward_discount = args.future_reward_discount
        self.experience_replay = args.experience_replay # Use prioritized experience replay?
        self.heuristic_bootstrap = args.heuristic_bootstrap # Use handcrafted grasping algorithm when grasping fails too many times in a row?
        self.explore_rate_decay = args.explore_rate_decay
        self.grasp_only = args.grasp_only

        # -------------- Testing options --------------
        self.is_testing = args.is_testing
        self.max_test_trials = args.max_test_trials # Maximum number of test runs per case/scenario
        self.test_preset_cases = args.test_preset_cases
        self.test_preset_file = os.path.abspath(args.test_preset_file) if self.test_preset_cases else None

        # ------ Pre-loading and logging options ------
        self.load_snapshot = args.load_snapshot # Load pre-trained snapshot of model?
        self.snapshot_file = os.path.abspath(args.snapshot_file)  if self.load_snapshot else None
        self.continue_logging = args.continue_logging # Continue logging from previous session
        self.logging_directory = os.path.abspath(args.logging_directory) if self.continue_logging else os.path.abspath('logs')
        self.save_visualizations = args.save_visualizations # Save visualizations of FCN predictions? Takes 0.6s per training step if set to True

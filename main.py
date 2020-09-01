import argparse
from utils.config import Config
from trainer.stage2trainer import Trainer as Stage2Trainer


class Solver():

	def __init__(self, args):
		self.robot_args = [args.obj_mesh_dir, args.num_obj, args.workspace_limits, args.heightmap_resolution]
		self.logger_args = {
			'continue_logging': args.continue_logging,
			'logging_directory': args.logging_directory,
			'workspace_limits': args.workspace_limits,
			'heightmap_resolution': args.heightmap_resolution,
			'continue_logging': args.continue_logging,
		}
		self.stage_2_trainer = Stage2Trainer(self.robot_args, self.logger_args)

	def main(self):
		self.stage_2_trainer.main()


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description='Train robotic agents to learn visual language grasp.'
	)
	# Run main program with specified config file
	parser.add_argument('-f', '--file', dest='file')
	args = parser.parse_args()
	solver = Solver(Config(args.file))
	solver.main()
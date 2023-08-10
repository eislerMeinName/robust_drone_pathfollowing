import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import argparse
from robust_drone_pathfollowing.spDRL.util.parameter_parser import parse_parameters
from robust_drone_pathfollowing.spDRL.experiments import WindSingleAgentExperiment
from gym_pybullet_drones.envs import WindSingleAgentAviary


def main():
    parser = argparse.ArgumentParser("Self-Paced Learning experiment runner")
    parser.add_argument("--base_log_dir", type=str, default="logs")
    parser.add_argument("--type", type=str, default="self_paced",
                        choices=["self_paced", "self_paced_v2"])
    parser.add_argument("--learner", type=str, choices=["ppo", "sac"])
    parser.add_argument("--env", type=str, default="point_mass",
                        choices=["windsingleagent"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--true_rewards", action="store_true", default=False)

    args, remainder = parser.parse_known_args()
    parameters = parse_parameters(remainder)

    if args.type == "self_paced":
        import torch
        torch.set_num_threads(1)
    args.learner = 'sac'

    exp = WindSingleAgentExperiment(args.base_log_dir, args.type, args.learner, parameters,
                                    args.seed, use_true_rew=args.true_rewards)

    exp.train()
    exp.evaluate()


if __name__ == "__main__":
    main()
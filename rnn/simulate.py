import numpy as np
import random
import torch
import warnings
warnings.filterwarnings('ignore')

from modules import *


if __name__ == '__main__':

    # set random seed
    seed = 15
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # parse args
    parser = ArgParser()
    args = parser.args

    # set experiment path
    exp_path = os.path.join(args.path, f'exp_{args.init_num_items}_{args.jobid}')

    # load net
    net = torch.load(os.path.join(exp_path, f'net.pth'), weights_only = False)

    # set environment
    env = MetaLearningWrapper(
        MemoryAcceptLeaveEnv(
            num_slots = args.num_slots,
            num_items = args.num_items,
            num_features = args.num_features,
            value_min = args.value_min,
            value_max = args.value_max,
            reward_std = args.reward_std,
            relevance_prob = args.relevance_prob,
            init_num_items = args.init_num_items,
            t_max = args.t_max,
            stay_cost = args.stay_cost,
            saccade_cost = args.saccade_cost,
            scale_factor = args.scale_factor,
            mode = 'exp',
        )
    )

    # simulate
    num_trials = 100000
    data = simulate(
        net = net,
        env = env,
        num_trials = num_trials,
        greedy = False,
        include_hidden = False,
    )
    save_data(data, os.path.join(exp_path, f'data_simulation.p'))



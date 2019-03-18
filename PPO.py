from __future__ import print_function

import argparse
import os
import sys
import pickle as pkl

import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from envs import create_atari_env
from model import ActorCritic
from train import train
from test import test
import my_optim
import pdb


if __name__ == '__main__':
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    env = create_atari_env(args.env_name)

    shared_model = ActorCritic(
        env.observation_space.shape[0], env.action_space)

    shared_model.share_memory()

    if args.no_shared:
        optimizer = None
    else:
        optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=args.lr)
        optimizer.share_memory()

    if args.task == 'train':
        processes = []

        p = mp.Process(target=test, args=(args.num_processes, args, shared_model))
        p.start()
        processes.append(p)
        for rank in range(0, args.num_processes):
            p = mp.Process(target=train, args=(rank, args, shared_model, optimizer))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    elif args.task == 'eval':
        shared_model.load_state_dict(torch.load(args.load_ckpt))
        test(args.num_processes, args, shared_model)
    elif args.task == 'develop':
        train(0, args, shared_model, optimizer)
import gym
import isaacgym
import isaacgymenvs
import numpy as np
import torch

from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
from isaacgymenvs.utils.utils import set_np_formatting, set_seed


envs = isaacgymenvs.make(
    seed=0,
    task="Jackal",
    num_envs=10,
    sim_device="cuda:0",
    rl_device="cuda:0",
    graphics_device_id=0,
    headless=True,
    multi_gpu=False,
    virtual_screen_capture=False,
    force_render=False,
)

R, SS, D, INFO = [], [], [], []

s = envs.reset()

a = torch.zeros((10, 2), device="cuda:0")

for i in range(501):
    _, r, d, info = envs.step(a) # Step
    R.append(r.detach().cpu())
    SS.append(info['success'].detach().cpu())
    D.append(d.detach().cpu())
    INFO.append(info['time_outs'].detach().cpu())

R  = torch.stack(R)
SS = torch.stack(SS)
D  = torch.stack(D)
INFO = torch.stack(INFO)
print(INFO.float()[-3:])
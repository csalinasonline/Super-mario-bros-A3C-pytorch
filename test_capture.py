"""
@author: Conrad Salinas <csalinasonline@gmail.com>
"""

import os
import time
import serial
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import torch
import numpy as np
from src.env import create_train_env
from src.model import ActorCritic
import torch.nn.functional as F
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of model described in the paper: Asynchronous Methods for Deep Reinforcement Learning for Super Mario Bros""")
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--action_type", type=str, default="complex")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--output_path", type=str, default="output")
    args = parser.parse_args()
    return args


def test(opt):
    # setup serial
    ser = serial.Serial(
      port='/dev/ttyACM0',
      baudrate=9600,
      parity=serial.PARITY_ODD,
      stopbits=serial.STOPBITS_TWO,
      bytesize=serial.SEVENBITS
    )
    if(ser.isOpen() == False):
      ser.open()
    else:
      ser.close()
      time.sleep(1)
      ser.open()
      time.sleep(1)
    #ser.close()
    # seed
    torch.manual_seed(123)
    # setup env
    env, num_states, num_actions = create_train_env(opt.world, opt.stage, opt.action_type,
                                                    "{}/video_{}_{}.mp4".format(opt.output_path, opt.world, opt.stage)) 
    #
    print(opt)   
    # constants
    num_states = 4
    num_actions = 12

    # load model and cuda
    model = ActorCritic(num_states, num_actions)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load("{}/a3c_super_mario_bros_{}_{}".format(opt.saved_path, opt.world, opt.stage)))
        model.cuda()
    else:
        model.load_state_dict(torch.load("{}/a3c_super_mario_bros_{}_{}".format(opt.saved_path, opt.world, opt.stage),
                                         map_location=lambda storage, loc: storage))
    #
    model.eval()
    # get inital state from start of Mario Stage via Image
    state = torch.from_numpy(env.reset())
    print(state.shape)
    time.sleep(2)
    done = True

    # loop
    while True:
        # 1st iter
        if done:
            h_0 = torch.zeros((1, 512), dtype=torch.float)
            c_0 = torch.zeros((1, 512), dtype=torch.float)
            env.reset()
        else:
            h_0 = h_0.detach()
            c_0 = c_0.detach()
        if torch.cuda.is_available():
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()
            state = state.cuda()
        # update via model output
        logits, value, h_0, c_0 = model(state, h_0, c_0)
        # get policy and action
        policy = F.softmax(logits, dim=1)
        action = torch.argmax(policy).item()
        action = int(action)
        msg = str(action + 1) + '\n'
        msg = msg.encode('utf_8')
        ser.write(msg)
        print('{0:02}'.format(action) + ':' + '{0:08b}'.format(action) + ':' + str(COMPLEX_MOVEMENT[action]))
        # update state
        state, reward, done, info = env.step(action)
        state = torch.from_numpy(state)
    #     #print(state)
    #     # render scene
        env.render()
        time.sleep(0.2)
        if info["flag_get"]:
            #print("World {} stage {} completed".format(opt.world, opt.stage))
            break
    ser.close()

if __name__ == "__main__":
    opt = get_args()
    test(opt)

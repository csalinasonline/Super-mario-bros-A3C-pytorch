"""
@author: Conrad Salinas <csalinasonline@gmail.com>
"""

import cv2
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
import matplotlib.pyplot as plt
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
#
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

#
def convert_state_to_img(input_state):
    #print(input_state.shape)
    state_2 = np.squeeze(input_state)
    #print(state_2.shape)
    state_3 = state_2[0:3,:,:].permute(1, 2, 0)
    state_4 = state_3 * 255.
    #print(state_3.shape)
    cv2.imshow('Input State to Img', np.array(state_4, dtype = np.uint8 ))

#
def test(opt):
    # setup serial
    ser = serial.Serial(
      port='/dev/ttyACM0',
      baudrate=9600,
      parity=serial.PARITY_ODD,
      stopbits=serial.STOPBITS_TWO,
      bytesize=serial.SEVENBITS
    )

    # check if serial is open or not?
    if(ser.isOpen() == False):
      ser.open()
    else:
      ser.close()
      time.sleep(1)
      ser.open()
      time.sleep(1)

    # seed
    torch.manual_seed(123)

    # setup env
    env, num_states, num_actions = create_train_env(opt.world, opt.stage, opt.action_type,
                                                    "{}/video_{}_{}.mp4".format(opt.output_path, opt.world, opt.stage)) 
    # what are the options?
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

    # eval
    model.eval()

    # get inital state from start of Mario Stage via Image
    state = torch.from_numpy(env.reset())
    convert_state_to_img(state)
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

        # send action to arduino nes controller
        msg = str(action + 1) + '\n'
        msg = msg.encode('utf_8')
        ser.write(msg)
        print('{0:02}'.format(action) + ':' + '{0:08b}'.format(action) + ':' + str(COMPLEX_MOVEMENT[action]))

        # update state
        state, reward, done, info = env.step(action)
        state = torch.from_numpy(state)

        # numpy to tensor
        convert_state_to_img(state)

        # show game
        env.render()

        # give some delay
        time.sleep(1)

        # finsihed level
        if info["flag_get"]:
            #print("World {} stage {} completed".format(opt.world, opt.stage))
            break

        # cv2 waits for a user input to quit the application
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # close serial
    ser.close()
    # close cv2
    cv2.destroyAllWindows()

if __name__ == "__main__":
    opt = get_args()
    test(opt)

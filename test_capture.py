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
import NesInterface as nes
from NesInterface import nes_button
from src.env import create_train_env
from src.model import ActorCritic
import torch.nn.functional as F
import matplotlib.pyplot as plt
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY

CONST_CAP_RES_WIDTH = 640
CONST_CAP_RES_HEIGHT = 480
CONST_NES_RES_WIDTH = 256
CONST_NES_RES_HEIGHT = 240
CONST_OFFSET_RES_WIDTH = (20, 476)
CONST_OFFSET_RES_HEIGHT = (128, 512)
CONST_FEATURE_RES_WIDTH = 84
CONST_FEATURE_RES_HEIGHT = 84

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

# method that reads the score from capture
def read_score():
    pass

# method that reads the coins from capture
def read_coins():
    pass

# method that reads the times from capture
def read_time():
    pass

# method that reads the world from capture
def read_world():
    pass

# method that reports flag goal from capture
def is_flag_goal():
    pass

# method that reports mario died from capture
def is_mario_dead():
    pass

# method that reports mario size from capture
def is_mario_size():
    pass

# method that reports mario power up type from capture
def is_mario_power_up():
    pass

# method that reports game over from capture
def is_game_over():
    pass

# methond go to main menu
def goto_main_menu():
    pass

# method to start lvl 1_1
def goto_lvl_1_1():
    pass

# method capture state
def capture_state():
    pass

# method reduce state
def reduce_state():
    pass

# method capture reset
def capture_reset():
    goto_main_menu()
    goto_lvl_1_1()

# method capture step
def capture_step():
    return 0, 0, 0, 0

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

    # open vid capture device
    cap = cv2.VideoCapture(0)

    # Check whether user selected camera is opened successfully.
    if not cap.isOpened():
      print("Could not open video device")

    # to set the resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # setup nes to main menu if not already (assume already console turned on)
    print(f'Reseting Nes to main menu')
    nes_button(ser, nes.NES_RESET)
    time.sleep(5)
    print(f'Start Mario Bros...')
    nes_button(ser, nes.NES_START)
    time.sleep(5)
    print(f'Start Game...')
    nes_button(ser, nes.NES_START)
    time.sleep(5)

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
    # MODIFIED
    ser.write('\n'.encode('utf_8'))
    capture_reset()
    #state = capture_state()
    #state = reduce_state()
    #state = torch.from_numpy(state)
    #convert_state_to_img(state)

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
        #ser.write(msg)
        print('{0:02}'.format(action) + ':' + '{0:08b}'.format(action) + ':' + str(COMPLEX_MOVEMENT[action]))

        # update state
        state, reward, done, info = env.step(action)
        # numpy to tensor
        state = torch.from_numpy(state)
        # MODIFIED
        #state, reward, done, info = capture_step()
        #state = torch.from_numpy(state)

        # capture nes frame-by-frame
        ret, frame = cap.read()
        # display the resulting frame
        cv2.imshow('nes full preview', frame)

        # convert nes frame to input feature
        #
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #
        frame_2 = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        #
        frame_3 = frame_2[20:476, 128:512]
        #
        frame_4 = cv2.resize(frame_3, (CONST_NES_RES_WIDTH, CONST_NES_RES_HEIGHT))
        #
        frame_5 = cv2.resize(frame_4, (CONST_FEATURE_RES_WIDTH, CONST_FEATURE_RES_HEIGHT))

        # display nes frame as feature
        cv2.imshow('nes feature preview', frame_5)

        # show state as img
        convert_state_to_img(state)

        # show game
        env.render()
        # give some delay
        time.sleep(1)
        # MODIFIED

        # finsihed level
        if info["flag_get"]:
            #print("World {} stage {} completed".format(opt.world, opt.stage))
            break
        # MODIFIED
        #f_goal = is_flag_goal()
        #if f_goal:
        #    break

        # cv2 waits for a user input to quit the application
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # clear nes ctrl
    ser.write('\n'.encode('utf_8'))
    # close serial
    ser.close()
    # close cv2
    cv2.destroyAllWindows()
    # release the capture
    cap.release()

if __name__ == "__main__":
    opt = get_args()
    test(opt)


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

CONST_DIM = 84
CONST_CAP_RES_WIDTH = 640
CONST_CAP_RES_HEIGHT = 480
CONST_NES_RES_WIDTH = 256
CONST_NES_RES_HEIGHT = 240
CONST_OFFSET_RES_WIDTH = (20, 478)
CONST_OFFSET_RES_HEIGHT = (102, 539)
CONST_FEATURE_RES_WIDTH = CONST_DIM
CONST_FEATURE_RES_HEIGHT = CONST_DIM


def calculate_cdf(histogram):
    """
    This method calculates the cumulative distribution function
    :param array histogram: The values of the histogram
    :return: normalized_cdf: The normalized cumulative distribution function
    :rtype: array
    """
    # Get the cumulative sum of the elements
    cdf = histogram.cumsum()
 
    # Normalize the cdf
    normalized_cdf = cdf / float(cdf.max())
 
    return normalized_cdf
 
def calculate_lookup(src_cdf, ref_cdf):
    """
    This method creates the lookup table
    :param array src_cdf: The cdf for the source image
    :param array ref_cdf: The cdf for the reference image
    :return: lookup_table: The lookup table
    :rtype: array
    """
    lookup_table = np.zeros(256)
    lookup_val = 0
    for src_pixel_val in range(len(src_cdf)):
        lookup_val
        for ref_pixel_val in range(len(ref_cdf)):
            if ref_cdf[ref_pixel_val] >= src_cdf[src_pixel_val]:
                lookup_val = ref_pixel_val
                break
        lookup_table[src_pixel_val] = lookup_val
    return lookup_table
 
def match_histograms(src_image, ref_image):
    """
    This method matches the source image histogram to the
    reference signal
    :param image src_image: The original source image
    :param image  ref_image: The reference image
    :return: image_after_matching
    :rtype: image (array)
    """
    # # Split the images into the different color channels
    # # b means blue, g means green and r means red
    # src_b, src_g, src_r = cv2.split(src_image)
    # ref_b, ref_g, ref_r = cv2.split(ref_image)
 
    # # Compute the b, g, and r histograms separately
    # # The flatten() Numpy method returns a copy of the array c
    # # collapsed into one dimension.
    # src_hist_blue, bin_0 = np.histogram(src_b.flatten(), 256, [0,256])
    # src_hist_green, bin_1 = np.histogram(src_g.flatten(), 256, [0,256])
    # src_hist_red, bin_2 = np.histogram(src_r.flatten(), 256, [0,256])    
    # ref_hist_blue, bin_3 = np.histogram(ref_b.flatten(), 256, [0,256])    
    # ref_hist_green, bin_4 = np.histogram(ref_g.flatten(), 256, [0,256])
    # ref_hist_red, bin_5 = np.histogram(ref_r.flatten(), 256, [0,256])

    src_hist, bin_0 = np.histogram(src_image.flatten(), 256, [0,256]) 
    ref_hist, bin_1 = np.histogram(ref_image.flatten(), 256, [0,256])      
 
    # Compute the normalized cdf for the source and reference image
    # src_cdf_blue = calculate_cdf(src_hist_blue)
    # src_cdf_green = calculate_cdf(src_hist_green)
    # src_cdf_red = calculate_cdf(src_hist_red)
    # ref_cdf_blue = calculate_cdf(ref_hist_blue)
    # ref_cdf_green = calculate_cdf(ref_hist_green)
    # ref_cdf_red = calculate_cdf(ref_hist_red)

    src_cdf = calculate_cdf(src_hist)
    ref_cdf = calculate_cdf(ref_hist)

    # # # Make a separate lookup table for each color
    # # blue_lookup_table = calculate_lookup(src_cdf_blue, ref_cdf_blue)
    # # green_lookup_table = calculate_lookup(src_cdf_green, ref_cdf_green)
    # # red_lookup_table = calculate_lookup(src_cdf_red, ref_cdf_red)

    lookup_table = calculate_lookup(src_cdf, ref_cdf)
 
    # # Use the lookup function to transform the colors of the original
    # # source image
    # blue_after_transform = cv2.LUT(src_b, blue_lookup_table)
    # green_after_transform = cv2.LUT(src_g, green_lookup_table)
    # red_after_transform = cv2.LUT(src_r, red_lookup_table)
    after_transform = cv2.LUT(src_image, lookup_table)
 
    # Put the image back together
    # image_after_matching = cv2.merge([
    #     blue_after_transform, green_after_transform, red_after_transform])
    # image_after_matching = cv2.convertScaleAbs(image_after_matching)

    image_after_matching = after_transform
    image_after_matching = cv2.convertScaleAbs(image_after_matching)    
 
    return lookup_table, image_after_matching

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
    cv2.imwrite('sim_img_1.png', np.array(state_4, dtype = np.uint8 ))
    state_5 = np.squeeze(np.array(state_4[:,:,0,None], dtype = np.uint8 ))
    return state_5

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
      baudrate=1000000,
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

    # load template - detect flag pole
    template = cv2.imread('Template.png',0)
    w, h = template.shape[::-1]

    # load template - detect gameover
    template_2 = cv2.imread('Template2.png',0)
    w_2, h_2 = template_2.shape[::-1]

    # setup nes to main menu if not already (assume already console turned on)
    print(f'Reseting Nes to main menu')
    nes_button(ser, nes.NES_RESET)
    time.sleep(2)
    print(f'Start Mario Bros...')
    nes_button(ser, nes.NES_START)
    time.sleep(2)
    print(f'Start Game...')
    nes_button(ser, nes.NES_START)
    time.sleep(2)

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
    st = env.reset()
    #print(st.shape)
    #print(type(st))
    # [1,4,84,84]
    state2 = torch.from_numpy(st)
    #print(state.shape)
    #print(type(state))
    img_1 = convert_state_to_img(state2)
    img_1b = img_1
    print(img_1)
    print(img_1.shape)
    time.sleep(2)

    # MODIFIED
    ser.write('\n'.encode('utf_8'))
    capture_reset()
    #state = capture_state()
    #state = reduce_state()
    #state = torch.from_numpy(state)
    #convert_state_to_img(state)

    N = 4
    offset = 15
    img_2 = np.zeros((CONST_DIM,CONST_DIM))

    # get 4 frames
    a = np.zeros((1,N,CONST_DIM,CONST_DIM))
    b = np.zeros((CONST_DIM,CONST_DIM), dtype = np.uint8)
    c = np.zeros((473, 437), dtype = np.uint8)
    b[:] = (offset)
    for i in range(N):
        # capture nes frame-by-frame
        ret, frame = cap.read()
        # convert nes frame to input feature
        frame_2 = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        #
        frame_3 = frame_2[CONST_OFFSET_RES_WIDTH[0]:CONST_OFFSET_RES_WIDTH[1], CONST_OFFSET_RES_HEIGHT[0]:CONST_OFFSET_RES_HEIGHT[1]]
        #
        c[0:15] = (157)
        c[15:473:,:] = frame_3
        #print(frame_3.shape)
        #
        frame_4 = cv2.resize(c, (CONST_NES_RES_WIDTH, CONST_NES_RES_HEIGHT))
        #
        frame_5 = cv2.resize(frame_4, (CONST_FEATURE_RES_WIDTH, CONST_FEATURE_RES_HEIGHT))
        #
        #frame_5 = cv2.add(frame_5, b)
        st = frame_5
        img_2 = frame_5
        img_2b = frame_5
        cv2.imwrite('nes_img_1.png', frame_5)
        a[0,i,...] = st
        print(f'{i}:{a.shape}')


    hist_lut, output_image = match_histograms(img_1b, img_2b)
    cv2.imwrite('nes_cor_img_1.png', output_image)

    print(img_2)
    print(img_2.shape)
    time.sleep(2)

    a = torch.from_numpy(a)
    a = a.float()

    img_2 = torch.from_numpy(img_2)
    img_2 = img_2.float()



    #a2 = np.squeeze(a)
    #print(state_2.shape)
    #a3 = a2[0:3,:,:].permute(1, 2, 0)
    #a4 = a3 * 255.
    #print(state_3.shape)
    #cv2.imshow('Input State to Img', np.array(a3, dtype = np.uint8 ))
    #print('Showing  1st nes frame')

    state = a

    done = True

    # loop
    while True:
        # 1st iter
        if done:
            h_0 = torch.zeros((1, 512), dtype=torch.float)
            c_0 = torch.zeros((1, 512), dtype=torch.float)
            #env.reset()
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
        #state, reward, done, info = env.step(action)
        #print(st.shape)
        # numpy to tensor
        #state = torch.from_numpy(state)
        #print(st.shape)
        # MODIFIED
        #state, reward, done, info = capture_step()
        #state = torch.from_numpy(state)

        # get 4 frames
        a = np.zeros((1,N,CONST_DIM,CONST_DIM))
        for i in range(N):
            # capture nes frame-by-frame
            ret, frame = cap.read()
            # display the resulting frame
            cv2.imshow('nes full preview', frame)

            # convert # frame to input feature
            frame_2 = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) #RGB, BGR, GBR, GRB
            #
            frame_3 = frame_2[CONST_OFFSET_RES_WIDTH[0]:CONST_OFFSET_RES_WIDTH[1], CONST_OFFSET_RES_HEIGHT[0]:CONST_OFFSET_RES_HEIGHT[1]]
            #
            c[0:15] = (157)
            c[15:473:,:] = frame_3
            #print(frame_3.shape)     
            #       
            frame_4 = cv2.resize(frame_3, (CONST_NES_RES_WIDTH, CONST_NES_RES_HEIGHT))
            #
            img = frame_4.copy()
            # Apply template Matching
            res = cv2.matchTemplate(img,template,5)
            threshold = 0.90
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            # Apply template Matching
            res_2 = cv2.matchTemplate(img,template_2,5)
            threshold_2 = 0.60
            min_val_2, max_val_2, min_loc_2, max_loc_2 = cv2.minMaxLoc(res_2)
            #
            frame_5 = cv2.resize(frame_4, (CONST_FEATURE_RES_WIDTH, CONST_FEATURE_RES_HEIGHT))
            #
            #frame_5 = cv2.add(frame_5, b)
            #
            after_transform = cv2.LUT(frame_5, hist_lut)
            frame_6 = cv2.convertScaleAbs(after_transform)             
            #
            st = frame_6
            a[0,i,...] = st
            #print(f'{i}:{a.shape}')

        a = torch.from_numpy(a)
        a = a.float()

        #print(frame_5.shape)
        #time.sleep(10)
        #
        state = a

        #a2 = np.squeeze(a)
        #print(state_2.shape)
        #a3 = a2[0:3,:,:].permute(1, 2, 0)
        #a4 = a3 * 255.
        #print(state_3.shape)
        #cv2.imshow('Input State to Img', np.array(a3, dtype = np.uint8 ))
        #print('Showing  1st nes frame')

        # numpy to tensor
        #state = torch.from_numpy(state)

        # display nes frame as feature
        #cv2.imshow('nes feature preview', frame_5)

        # show state as img
        #convert_state_to_img(state)

        # show game
        #env.render()
        # give some delay
        #time.sleep(1)
        # MODIFIED

        # finsihed level
        #if info["flag_get"]:
            #print("World {} stage {} completed".format(opt.world, opt.stage))
        #    break
        # MODIFIED
        #f_goal = is_flag_goal()
        #if f_goal:
        #    break
        if max_val > threshold:
            break
        #
        if max_val_2 > threshold_2:
            # setup nes to main menu if not already (assume already console turned on)
            print(f'Reseting Nes to main menu')
            nes_button(ser, nes.NES_RESET)
            time.sleep(2)
            print(f'Start Mario Bros...')
            nes_button(ser, nes.NES_START)
            time.sleep(2)
            print(f'Start Game...')
            nes_button(ser, nes.NES_START)
            time.sleep(2)        
        #

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

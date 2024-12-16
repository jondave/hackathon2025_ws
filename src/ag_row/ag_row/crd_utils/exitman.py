__author__ = "Rajitha de Silva"
__copyright__ = "Copyright (C) 2023 rajitha@ieee.org"
__license__ = "CC"
__version__ = "1.0"

import cv2
import os
import time
import math
import statistics
from geometry_msgs.msg import Twist

SIMILARITY_THRESHOLD = 20 # Similarity threshold for sift feature matching
EXIT_VELOCITY=0.1
FRAME_VIEW_DELAY=0.5
SCAN_LIMIT = 80
CAM_VFOV_ANGLE = 58

def move_odom(odomx,dis,start_posx, exit_velocity=EXIT_VELOCITY,):
    '''
    This function will move the robot a 'dis' distance along x-axis (forward for Husky) based on odometry of the robot
    ''' 
    move_cmd = Twist()
    exit_status = False

    if abs(odomx - start_posx) < dis:
        move_cmd.linear.x = exit_velocity
        #print("Elapsed Distance: ", abs(odomx - start_posx))
        exit_status = True   
    return move_cmd, exit_status

def exit_matcher(first_img_rgb, curr_img_rgb, similarity_threshold = SIMILARITY_THRESHOLD, exit_velocity=EXIT_VELOCITY, frame_view_delay=FRAME_VIEW_DELAY):
    '''
    This function will return velocity commands for the robot in order to exit a crop row.                   
    first_image_rgb: Image captured when EOR Detector determine the robot is near row end                    
    curr_img_rgb: Subsequent images captured during exit manoeuvre is running                                              
    similarity_threshold: Matching sift feature count threshold to decide that robot is precisely at EOR    
    exit_velocity: Forward velocity of the robot during exit manoeuvre                     
    frame_view_delay: How long to display the similarity matching image
    '''

    # Load the first image and extract SIFT features
    first_img = cv2.cvtColor(first_img_rgb, cv2.COLOR_BGR2GRAY)
    first_kp, first_desc = cv2.xfeatures2d.SIFT_create().detectAndCompute(first_img, None)
     
    # Load the current image and extract SIFT features
    curr_img = cv2.cvtColor(curr_img_rgb, cv2.COLOR_BGR2GRAY)
    curr_kp, curr_desc = cv2.xfeatures2d.SIFT_create().detectAndCompute(curr_img, None)

    # Match keypoints and filter out bad matches
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    matches = matcher.knnMatch(first_desc, curr_desc, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Compute similarity score based on number of good matches
    sim_score = len(good_matches)
    print(sim_score)    

    # Display the image if frame_view_delay is non-zero
    if frame_view_delay>0:
        # Visualize the feature matching and similarity score on the image
        img_with_matching = cv2.drawMatches(first_img_rgb, first_kp, curr_img_rgb, curr_kp, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        img_with_matching = cv2.putText(img_with_matching, f'Similarity score: {sim_score}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

        # Display the image with feature matching, similarity score overlay
        cv2.imshow('Image with feature matching overlay', img_with_matching)
        cv2.waitKey(1)

    move_cmd = Twist()
    stop_flag = True

    # Set the robot to move forward if the robot haven't reached EOR State
    if sim_score > similarity_threshold:
        move_cmd.linear.x = exit_velocity
        stop_flag = False        

    return move_cmd, stop_flag

def line_to_distance(line):
    '''
    This function will recieve a pixel row from a depth image and return the distance from the camera to that line on the ground
    '''
    dis = statistics.median(line)
    return dis

def image_to_dfov(depth_image):
    '''
    This function will recieve a depth image and return the ground distance within the field of view (dfov).
    The depth image is supposed to represent a flat swath of arable land within its field of view.
    Expected a single channel depth image. (h x w)
    '''
    # Extract Distances to far and near lines of the swath
    h, w = depth_image.shape
    top_row = depth_image[SCAN_LIMIT,:]
    bottom_row = depth_image[(h-SCAN_LIMIT),:]
    Dt = line_to_distance(top_row)
    Db = line_to_distance(bottom_row)

    # Use law of cosines to calculate ground distance
    dfov = math.sqrt((Dt*Dt)+(Db*Db)-(2*Dt*Db*math.cos(math.radians(CAM_VFOV_ANGLE))))
    print(Dt)
    print("dt")
    print(Db)
    print("db")

    cv2.imshow('Id overlay', depth_image)
    cv2.waitKey(1)
    return dfov

def dfov_to_cropped(image, robot_length, dfov):
    '''
    This function will reveive an image, length of the robot and a 'dfov' distance.
    It will return a cropped image proportional to a swath of field equivalent to the length of the robot
    '''
    h = image.shape[0]

    if dfov > robot_length:
        crop_limit = round((robot_length*h)/dfov)
        crop_limit = h - crop_limit
        crop_img = image[crop_limit:h, :]

    else:# Future work: Write a recursive algorithm to compensate for lower fov and still nivigate the desired distance.
        crop_img = image
        crop_limit = h
        print("WARNING: Exit stage 2 might stop early. Adjust Camera to Increase the Field of View!")
    return crop_img, crop_limit

def exitman(first_image_rgb, curr_image_rgb):
    '''
    This function will control the robot's velocity in 2 stages.                           |         ***EOR State***                           ***Exit State***
        Stage 1: Drive the robot from EOR Detection stage to EOR Position                  | ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾   ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾  |
        Stage 2: Drive the robot from EOR Position to Exit Position                        |                |‾\  EOR Position |           EOR Position   |‾\  Exit Position |
    first_image_rgb: Image captured when EOR Detector determine the robot is near row end  |               /‾‾‾\  |           |                     |   /‾‾‾\  |            |
    curr_img_rgb: Subsequent images captured during exit manoeuvre is running              |  __w_w_w_w_w_O‾‾‾‾‾O_↓           | __w_w_w_w_w_w_w_w_w_↓__O‾‾‾‾‾O_↓            |
    depth_img: Depth images captured during exit manoeuvre is running   
    eor_pos : Position of detected end of row                        
    '''

    exit_status = True # Initialize the manoeuvre status flag
    move_cmd, stop_flag = exit_matcher(first_image_rgb, curr_image_rgb)

    if stop_flag:
        exit_status = False
    return move_cmd, exit_status

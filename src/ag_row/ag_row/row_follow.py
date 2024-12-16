#!/usr/bin/env python3
__author__ = "Rajitha de Silva"
__copyright__ = "Copyright (C) 2023 rajitha@ieee.org"
__license__ = "CC"
__version__ = "2.0"

'''
Version 3 Change Log
=====================
> Removed Velocity Latch initialization
> Updated cmd_vel to be latched in ROS2
> Changed 'agrow' to 'ag_row'
'''

#from __future__ import print_function
import sys
import time
import subprocess
import random
import math
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
# import rospkg
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import img_to_array, array_to_img

from sensor_msgs.msg import Image as IGG
from sensor_msgs.msg import Imu, CameraInfo
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

from ag_row.crd_utils.unetwsess import * #U-Net model inference functions
from ag_row.crd_utils.data import * # Data Loader for U-Net
from ag_row.crd_utils.triangle_scan import * #CRD Post processing
from ag_row.crd_utils.exitman import * # Exit Manoeuvre
from ag_row.crd_utils.utils import * #Miscellaneous (math functions and etc)

ROBOT = "FIRA" # [Husky, Mark1, Hunter, Leo, HuskySim]
CAMERA = "FIRA"# [D435i, Leo]

class lineFollower(object):
    def __init__(self, robot, camera):
        rclpy.init()
        self.node = Node('line_follower')
        vel_topic, odom_topic, length = init_robot(robot)
        img_topic, depth_topic, imu_topic, cam_info_topic = init_camera(camera)

        self.rgb_image = None
        self.depth_image = None
        self.camera_info = None
        self.imu_data = None 
        self.odom = None
        self.cr_mask = None
        self.reentry = None
        self.eor = None
        self.robot_length = length

        self.node.create_subscription(IGG, img_topic, self.rgb_callback, 10)
        self.node.create_subscription(IGG, depth_topic, self.depth_callback, 10)
        self.node.create_subscription(Imu, imu_topic, self.imu_callback, 10)
        self.node.create_subscription(CameraInfo, cam_info_topic, self.camera_info_callback, 10)
        self.node.create_subscription(Odometry, odom_topic, self.odom_callback, 10)

        self.pub_vel = self.node.create_publisher(Twist, vel_topic, 10)
        self.pub_vui = self.node.create_publisher(Twist, "/vui", 10)#Vision User Interface Topic. An image verbose for each thread in the pipeline
        print("Line follower initialized")

        #self.run()


    # Callback Functions
    def rgb_callback(self, data):
        print("rgb_callback")
        img = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
        print(img)
        self.rgb_image = cv2.resize(img,(512,512))
        t1 = time.time()
        #Predict Crop Row Mask
        img = img_to_array(self.rgb_image)
        img = img.astype('float32')
        img /= 255
        img = tf.reshape(img, [1, 512, 512, 3])
        img_mask = model.predict(img, verbose=0)
        img = img_mask[0, :]
        img = np.array(array_to_img(img))
        binary, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        self.cr_mask = img
        #print("T: ",time.time()-t1,"ms")

    def depth_callback(self, data):
        print("depth_callback")
        depth = np.frombuffer(data.data, dtype=np.uint16).reshape(data.height, data.width)
        self.depth_image = cv2.resize(depth,(512,512))
        self.depth_image = self.depth_image/1000.0 # Multiply by [depth scale = 0.001] in mm

    def camera_info_callback(self, data):
        print("camera_info_callback")
        self.camera_info = data

    def imu_callback(self, data): 
        print("imu_callback")
        self.imu_data = data

    def odom_callback(self, data):
        print("odom_callback")
        self.odom = data

    # Operation Functions
    def tag_reentry(self, re_entry, eor):#Identify the Re-Entry Point in Next Row
        self.eor_image = self.rgb_image[eor:self.rgb_image.shape[0], :]
        self.reentry = re_entry
        self.eor = eor
        return 0

    def exit_row(self):#Exit Crop Row with Maneuvere
    #This function will control the robot's velocity in 2 stages.          |         ***EOR State***                           ***Exit State***
    #                                                                      | ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾   ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾  |
    #    Stage 1: Drive the robot from EOR Detection stage to EOR Position |                |‾\  EOR Position |           EOR Position   |‾\  Exit Position |
    #    Stage 2: Drive the robot from EOR Position to Exit Position       |               /‾‾‾\  |           |                     |   /‾‾‾\  |            |
    #                                                                      |  __w_w_w_w_w_O‾‾‾‾O_↓           | __w_w_w_w_w_w_w_w_w_↓__O‾‾‾‾O_↓            | 
        cv2.destroyAllWindows()                     
        #Stage 1: Drive the robot from EOR Detection stage to EOR Position 
        exit_status = True
        while exit_status:
            move_cmd, exit_status = exitman(self.eor_image, self.rgb_image)
            self.pub_vel.publish(move_cmd)
        print("Stage 1 Complete. Starting Stage 2")
        
        time.sleep(3)
        #Stage 2: Drive the robot from EOR Position to Exit Position  
        exit_status = True
        cv2.destroyAllWindows()   
        start_position = self.odom.pose.pose.position.x # record odometry position for given point
        print(self.odom.pose.pose.position.x)
        #dfov = image_to_dfov(self.depth_image)# Get dfov
        #self.eor_image, crop_limit= dfov_to_cropped(self.rgb_image, self.robot_length, dfov)# Update new EOR Image
        #self.eor_image = self.rgb_image[crop_limit:self.rgb_image.shape[0], :]
        while exit_status:
            #move_cmd, exit_status = exitman(self.eor_image, self.rgb_image[crop_limit:self.rgb_image.shape[0], :])
            move_cmd, exit_status = move_odom(self.odom.pose.pose.position.x,self.robot_length,start_position)
            self.pub_vel.publish(move_cmd)
        print("Stage 2 Complete")
        return 0

    def tri_scan(self):
        crop_scan = LineScan(self.cr_mask, self.rgb_image)
        center_line_angle, center_line_pos, exflg, re_entry, eor = crop_scan.scan()
        del crop_scan
        return center_line_angle, center_line_pos, exflg, re_entry, eor

    def reset(self):
        self.rgb_image = None
        self.depth_image = None
        self.camera_info = None
        self.imu_data = None 
        self.odom = None
        self.cr_mask = None
        self.reentry = None
        self.eor = None
        return 0

    def nav_diagnose(self):
        if self.rgb_image is None:
            print("RGB Image Not Received")
        if self.cr_mask is None:
            print("Crop Row Mask Not Received")
        if self.depth_image is None:
            print("Depth Image Not Received")
        if self.camera_info is None:
            print("Camera Info Not Received")
        if self.odom is None:
            print("Odometry Not Received")
        if self.imu_data is None:
            print("IMU Data Not Received")
        return 0

    # Main Algorithm 
    def run(self):
        rate = self.node.create_rate(100)

        while rclpy.ok():
            if self.rgb_image is not None and self.cr_mask is not None and self.depth_image is not None and self.camera_info is not None:
                print("$$$$$$")
                depth_scale = self.camera_info.K[0]  # Get the depth scale from the camera info
                depth_image_meters = self.depth_image * depth_scale

                center_line_angle, center_line_pos, exflg, re_entry, eor = self.tri_scan()
                cerror = center_line_angle # cw + | ccw -
                derror = 256 - center_line_pos # Right pos - | Left pos +

                if exflg:#if exit flag is set, perform exit maneuvere
                    self.tag_reentry(re_entry, eor)#Record Re-Entry Position
                    self.exit_row()#Exit Crop Row
                    rclpy.shutdown()

                #Control Robot
                move_cmd = Twist()
                move_cmd.linear.x = 0.03
                move_cmd.angular.z = ((cerror/500) + (derror/3000)) # cw - | ccw + #Husky
                #move_cmd.angular.z = ((cerror/400) + (derror/2000)) # cw - | ccw + #Husky
                self.pub_vel.publish(move_cmd)

                # Reset the all parameters after processing
                self.reset()

            else:
                #continue
                print("Navigation failed! Insufficient data!")
                self.nav_diagnose()
            #rate.sleep()
        rclpy.spin()



def main():
    myunet = myUnet()
    model = myunet.get_unet()
    # model_path = get_package_share_directory("ag_row") + "/ag_row/models/unet.hdf5"
    model_path = "/home/corn/code/hackathon2025_ws/src/ag_row/ag_row/models/unet.hdf5"
    model.load_weights(model_path)
    print("U-Net Model Loaded Successfully")  

    line_follower = lineFollower(ROBOT, CAMERA)
    line_follower.run()

if __name__ == '__main__':
    main();


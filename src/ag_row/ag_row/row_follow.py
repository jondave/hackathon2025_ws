#!/usr/bin/env python3
__author__ = "Rajitha de Silva"
__license__ = "CC"
__version__ = "3.0"

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
from romea_mobile_base_msgs.msg import TwoAxleSteeringCommand
from romea_cmd_mux_msgs.srv import Unsubscribe
from romea_cmd_mux_msgs.srv import Subscribe

from ag_row.crd_utils.unetwsess import * #U-Net model inference functions
from ag_row.crd_utils.data import * # Data Loader for U-Net
from ag_row.crd_utils.triangle_scan import * #CRD Post processing
from ag_row.crd_utils.pid import * #PID Controller
from ag_row.crd_utils.exitman import * # Exit Manoeuvre
from ag_row.crd_utils.utils import * #Miscellaneous (math functions and etc)

ROBOT = "FIRA" # [Husky, Mark1, Hunter, Leo, HuskySim]
CAMERA = "FIRA"# [D435i, Leo]

## this class adds the new cmd_vel topic to the twist mux in the FIRA robot
# https://github.com/FiraHackathon/hackathon2025_ws/blob/main/doc/robot_control.md 
class CmdMuxServiceClient(Node):
    def __init__(self):
        super().__init__('cmd_mux_service_client')
        self.client = self.create_client(Subscribe, '/robot/base/cmd_mux/subscribe')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

    def call_subscribe_service(self):
        request = Subscribe.Request()
        request.topic = '/row_following_cmd_vel'
        request.priority = 50
        request.timeout = 0.1
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info(f'Successfully subscribed to {request.topic}')
        else:
            self.get_logger().error('Service call failed')

class lineFollower(Node):
    def __init__(self, robot, camera):
        super().__init__('Line_Follower')
        vel_topic, odom_topic, length, kp, ki, kd = init_robot(robot)
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
        self.pid = PIDController(kp,ki,kd,0.0)#Initialize PID with target error 0.0

        self.create_subscription(IGG, img_topic, self.rgb_callback, 10)
        #self.create_subscription(IGG, depth_topic, self.depth_callback, 10)
        self.create_subscription(Imu, imu_topic, self.imu_callback, 10)
        self.create_subscription(CameraInfo, cam_info_topic, self.camera_info_callback, 10)
        self.create_subscription(Odometry, odom_topic, self.odom_callback, 10)

        # self.pub_vel = self.create_publisher(TwoAxleSteeringCommand, vel_topic, 10)
        self.pub_vel = self.create_publisher(TwoAxleSteeringCommand, "/row_following_cmd_vel", 10)
        self.pub_vui = self.create_publisher(Twist, "/vui", 10)#Vision User Interface Topic. An image verbose for each thread in the pipeline
        print("Line follower initialized")

        self.model = self.load_unet_model()

        #self.run()


    # Callback Functions
    def rgb_callback(self, data):
        img = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
        left_margin = right_margin = 350   # Left, Right margins in pixels
        top_edge = 300  # The y-coordinate of the top edge of the cropping area (adjust as needed)
        cropped_image = img[top_edge:720, left_margin:img.shape[1] - right_margin]

        self.rgb_image = cv2.resize(cropped_image,(512,512))
        t1 = time.time()
        #Predict Crop Row Mask
        img = img_to_array(self.rgb_image)
        img = img.astype('float32')
        img /= 255
        img = tf.reshape(img, [1, 512, 512, 3])
        img_mask = self.model.predict(img, verbose=0)
        img = img_mask[0, :]
        img = np.array(array_to_img(img))
        binary, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        self.cr_mask = img
        print("T: ",time.time()-t1,"ms")

    def depth_callback(self, data):
        depth = np.frombuffer(data.data, dtype=np.uint16).reshape(data.height, data.width)
        self.depth_image = cv2.resize(depth,(512,512))
        self.depth_image = self.depth_image/1000.0 # Multiply by [depth scale = 0.001] in mm

    def camera_info_callback(self, data):
        self.camera_info = data

    def imu_callback(self, data): 
        self.imu_data = data

    def odom_callback(self, data):
        self.odom = data

    # Operation Functions
    def tag_reentry(self, re_entry, eor):#Identify the Re-Entry Point in Next Row
        self.eor_image = self.rgb_image[eor:self.rgb_image.shape[0], :]
        self.reentry = re_entry
        self.eor = eor
        return 0

    '''
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
        '''

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
    
    def load_unet_model(self):
        myunet = myUnet()
        model = myunet.get_unet()
        # model_path = get_package_share_directory("ag_row") + "/ag_row/models/unet.hdf5"
        model_path = "/home/corn/code/hackathon2025_ws/src/ag_row/ag_row/models/unet.hdf5"
        model.load_weights(model_path)
        print("U-Net Model Loaded Successfully")  

        return model

    # Main Algorithm 
    def run(self):
        #rate = self.node.create_rate(100)

        #while rclpy.ok():
        if self.rgb_image is not None and self.cr_mask is not None and self.camera_info is not None:
            print("$Runing Navigation Loop$")
            #depth_scale = self.camera_info.K[0]  # Get the depth scale from the camera info
            #depth_image_meters = self.depth_image * depth_scale

            center_line_angle, center_line_pos, exflg, re_entry, eor = self.tri_scan()
            a_error = -center_line_angle # cw + | ccw -
            d_error = center_line_pos - 256 # Right pos - | Left pos +

            #if exflg:#if exit flag is set, perform exit maneuvere
                #self.tag_reentry(re_entry, eor)#Record Re-Entry Position
                #self.exit_row()#Exit Crop Row
                #rclpy.shutdown()

            #Control Robot
            #move_cmd = Twist()
            #move_cmd.linear.x = 0.5
            #move_cmd.angular.z = ((cerror/500) + (derror/3000)) # cw - | ccw + #Husky
            #self.pub_vel.publish(move_cmd)

            combined_error = 0.95*a_error + 0.05*d_error
            steer_angle = self.pid.update(combined_error)

            move_cmd = TwoAxleSteeringCommand()
            move_cmd.longitudinal_speed = 0.3
            move_cmd.front_steering_angle = -steer_angle
            move_cmd.rear_steering_angle = steer_angle
            self.pub_vel.publish(move_cmd)

            # Reset the all parameters after processing
            self.reset()

        else:
            #continue
            print("Navigation failed! Insufficient data!")
            #self.nav_diagnose()
            #rate.sleep()
        #rclpy.spin()



def main(args=None):
    rclpy.init(args=args)

    ## this class adds the new cmd_vel topic to the twist mux in the FIRA robot
    # https://github.com/FiraHackathon/hackathon2025_ws/blob/main/doc/robot_control.md 
    client = CmdMuxServiceClient()
    client.call_subscribe_service()

    line_follower = lineFollower(ROBOT, CAMERA)
    #line_follower.run()
    try:
        while rclpy.ok():
            rclpy.spin_once(line_follower, timeout_sec=0.5)
            line_follower.run()
    except KeyboardInterrupt:
        line_follower.get_logger().info("Shutting Down Line Follower...")
    finally:
        line_follower.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main();


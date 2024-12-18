__author__ = "Rajitha de Silva"
__license__ = "CC"
__version__ = "1.0"

import math

def euler_from_quaternion(x, y, z, w):
    '''
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    '''
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
     
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
     
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z # in radians

def init_camera(name):
    '''
    Return the camera specific parameters given the camera name
    '''

    #Default
    img_topic = "/camera/color/image_raw"
    depth_topic = "/camera/depth/image_rect_raw"
    imu_topic = "/camera/imu"
    cam_info_topic = "/camera/color/camera_info"

    if name == "D435i":
        img_topic = "/camera/color/image_raw"
        depth_topic = "/camera/depth/image_rect_raw"
        imu_topic = "/camera/imu"
        cam_info_topic = "/camera/color/camera_info"

    if name == "Leo":
        img_topic = "/camera/image_raw"
        depth_topic = "/camera/image_raw" # *No depth Stearm
        imu_topic = "/imu"
        cam_info_topic = "/camera/camera_info"

    if name == "FIRA":
        img_topic = "/robot/rgbd_camera/rgb/image_raw"
        depth_topic = "/robot/rgbd_camera/depth/image_raw" # *No depth Stearm
        imu_topic = "/robot/imu/data"
        cam_info_topic = "/robot/rgbd_camera/depth/camera_info"

    return img_topic, depth_topic, imu_topic, cam_info_topic

def init_robot(name):
    '''
    Return the robot specific parameters given the robot name
    '''

    #Default
    vel_topic = "/rf_vel"
    odom_topic = "/odom"
    length = 0.526 # Meters

    if name == "Mark1":
        vel_topic = "/rf_vel"
        odom_topic = "/odom"
        length = 0.526 # Meters

    elif name == "Husky":
        vel_topic = "/rf_vel"
        odom_topic = "/odometry/filtered"
        length = 0.990 # Meters

    elif name == "FIRA":
        vel_topic = "/robot/base/teleop/cmd_two_axle_steering"
        #vel_topic = "rf_vel"
        odom_topic = "/robot/base/controller/odom"
        length = 0.990 # Meters

    return vel_topic, odom_topic, length



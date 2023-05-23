from __future__ import print_function
#Need to import MechEye Device first
from MechEye import Device

import rospy
import moveit_commander
import moveit_msgs.msg
from geometry_msgs.msg import PoseStamped, Pose 
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from tf.transformations import *
import tf2_ros

import numpy as np
import cv2
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R
import sys
import os
import copy
import time
from pathlib import Path
import colorsys


class MechCapture(object): 
    def __init__(self):
        self.device = Device()
        self.device.set_scan_2d_exposure_mode("Auto")
        self.find_camera_list()
        self.connect_device_info()
        self.array = []
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

    def show_error(self, status):
        if status.ok():
            return
        print("Error Code : {}".format(status.code()),
        ",Error Description: {}".format(status.description()))


    def print_device_info(self, num, info):
        print(" Mech-Eye device index: {}\n".format(str(num)),
            "Camera Model Name: {}\n".format(info.model()),
            "Camera ID: {}\n".format(info.id()),
            "Camera IP: {}\n".format(info.ip()),
            "Hardware Version: {}\n".format(info.hardware_version()),
            "Firmware Version: {}\n".format(info.firmware_version()),
            "...............................................")

    def find_camera_list(self):
        print("\nFind Mech-Eye devices...")
        self.device_list = self.device.get_device_list()
        if len(self.device_list) == 0:
            print("No Mech-Eye device found.")
            quit()
        for i, info in enumerate(self.device_list):
            self.print_device_info(i, info)

    def connect_device_info(self):
        # Automatically connect to the first camera
        status = self.device.connect(self.device_list[0])
        if not status.ok():
            self.show_error(status)
            quit()
        print("Connected to the Mech-Eye device successfully.")

    def capture_color_image(self, counter, path):
        color_map = self.device.capture_color()
        this_img_filename = str(path.joinpath(f"color/{counter:03}.png")) # the :02 is to represent number of digits

        # Saving colour image into file
        cv2.imwrite(this_img_filename, color_map.data())
        self.get_mechmind_tf(path, counter)
        print(f"Saved mechmind_color{counter}.png")

    def capture_depth(self, counter, path):
        depth_map = self.device.capture_depth()
        this_img_filename = str(path.joinpath(f"depth/{counter:03}.png")) # the :02 is to represent number of digits
        cv2.imwrite(this_img_filename, depth_map.data().astype(np.uint16))
        print(f"Saved mechmind_depth{counter}.png")

    def get_mechmind_tf(self, path_npy_folder, counter):
        try:
            trans = self.tfBuffer.lookup_transform("base_link","mechmind_camera/color_map",  rospy.Time(0))

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as ex:
            rospy.logwarn(str(ex))
            rospy.Rate(10.0).sleep()
            return
        pose_msg = []
        pose_msg.append(trans.transform.translation.x)
        pose_msg.append(trans.transform.translation.y)
        pose_msg.append(trans.transform.translation.z)
        pose_msg.append(trans.transform.rotation.x)
        pose_msg.append(trans.transform.rotation.y)
        pose_msg.append(trans.transform.rotation.z)
        pose_msg.append(trans.transform.rotation.w)
        new_matrix = self.get_transform_mat_from_pose(pose_msg)
        self.array.append(new_matrix)
        np.array(self.array)
        npy_filename = str(path_npy_folder.joinpath(f"tf_mechmind_to_baselink"))
        np.save(npy_filename, self.array)

    def get_transform_mat_from_pose(self, transform):
        """
        get transform matrix from ROS pose
        transform matrix: [4*4]
        pose: [7], (translation [3], orientation [4])
        """
        transform_mat = np.zeros(shape=(4,4))  
        transform_mat[0:3,0:3] = R.from_quat([transform[3], transform[4], transform[5], transform[6]]).as_matrix()
        transform_mat[0][3] = transform[0]
        transform_mat[1][3] = transform[1]
        transform_mat[2][3] = transform[2]
        transform_mat[3][3] = 1

        return transform_mat
    
    def disconnect(self):
        self.device.disconnect()
        print("Disconnected from the Mech-Eye device successfully.")

class RealsenseCapture(object):
    def __init__(self):
        self.br = CvBridge()
        self.pose_array = []
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

    def get_realsense_tf(self, path_npy_folder):
        try:
            trans = self.tfBuffer.lookup_transform("base_link",  "realsense_camera", rospy.Time(0))

            pose_msg = []
            pose_msg.append(trans.transform.translation.x)
            pose_msg.append(trans.transform.translation.y)
            pose_msg.append(trans.transform.translation.z)
            pose_msg.append(trans.transform.rotation.x)
            pose_msg.append(trans.transform.rotation.y)
            pose_msg.append(trans.transform.rotation.z)
            pose_msg.append(trans.transform.rotation.w)
            new_matrix = self.get_transform_mat_from_pose(pose_msg)
            self.pose_array.append(new_matrix)
            np.array(self.pose_array)
            npy_filename = str(path_npy_folder.joinpath(f"tf_realsense_to_baselink"))
            np.save(npy_filename, self.pose_array)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as ex:
            rospy.logwarn(str(ex))
            pass


    def get_transform_mat_from_pose(self, transform):
        """
        get transform matrix from ROS pose
        transform matrix: [4*4]
        pose: [7], (translation [3], orientation [4])
        """
        transform_mat = np.zeros(shape=(4,4))  
        transform_mat[0:3,0:3] = R.from_quat([transform[3], transform[4], transform[5], transform[6]]).as_matrix()
        transform_mat[0][3] = transform[0]
        transform_mat[1][3] = transform[1]
        transform_mat[2][3] = transform[2]
        transform_mat[3][3] = 1

        return transform_mat

    def capture_realsense_color(self, counter, path):
        this_img_filename = str(path.joinpath(f"color/{counter:03}.png")) # the :02 is to represent number of digits

        # Saving colour image into file
        data = rospy.wait_for_message("/camera/color/image_raw", Image)
        img_color = self.br.imgmsg_to_cv2(data, desired_encoding='bgr8') 
        cv2.imwrite(this_img_filename, img_color)
        self.get_realsense_tf(path)

        print(f"Saved realsense_color{counter}.png")

    def capture_realsense_depth(self, counter, path):
        this_img_filename = str(path.joinpath(f"depth/{counter:03}.png")) # the :02 is to represent number of digits
        # Saving colour image into file
        data = rospy.wait_for_message("/camera/aligned_depth_to_color/image_raw", Image)
        img_depth = self.br.imgmsg_to_cv2(data, desired_encoding='passthrough')
        cv2.imwrite(this_img_filename, img_depth.astype(np.uint16))
        print(f"Saved realsense_depth{counter}.png")

    def lookup_transform(self, from_here, to_here):
        try:
            tfBuffer = tf2_ros.Buffer()
            listener = tf2_ros.TransformListener(tfBuffer)
            camlink2cambase = tfBuffer.lookup_transform(from_here, to_here, rospy.Time(0))
            camlink2cambase_pose = []
            camlink2cambase_pose.append(camlink2cambase.transform.translation.x)
            camlink2cambase_pose.append(camlink2cambase.transform.translation.y)
            camlink2cambase_pose.append(camlink2cambase.transform.translation.z)
            camlink2cambase_pose.append(camlink2cambase.transform.rotation.x)
            camlink2cambase_pose.append(camlink2cambase.transform.rotation.y)
            camlink2cambase_pose.append(camlink2cambase.transform.rotation.z)
            camlink2cambase_pose.append(camlink2cambase.transform.rotation.w)
            return camlink2cambase_pose

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException)as ex:
            rospy.logwarn(str(ex))
            rospy.Rate(10).sleep()
            pass


    def get_pose_from_transform_mat(self, transform_mat):
        """
        get ROS pose from transform matrix
        pose: [7], (translation [3], orientation [4])
        transform matrix: [4*4]
        """
        quat = R.from_matrix(transform_mat[0:3,0:3]).as_quat()
        return [transform_mat[0,3], transform_mat[1,3], transform_mat[2,3], quat[0], quat[1], quat[2], quat[3]]
        

class MoveGroupPythonInterface(object):

    def __init__(self):
        #Initialise Robot and Scene
        super(MoveGroupPythonInterface, self).__init__()
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group_name = "manipulator"
        self.move_group = moveit_commander.MoveGroupCommander(self.group_name)
        self.display_trajectory_publisher = rospy.Publisher(
            "/move_group/display_planned_path",
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=20,
        )
        self.planning_frame = self.move_group.get_planning_frame()
        self.eef_link = self.move_group.get_end_effector_link()
        self.group_names = self.robot.get_group_names()
        self.box_name = " "

        #Create Folder
        self.dir = Path(__file__).parents[0].joinpath("stored_data_mechmind")
        self.scene_name = self.create_next_folder()
        self.path_image_folder_m = Path(__file__).parents[0].joinpath("stored_data_mechmind", self.scene_name)
        self.path_image_folder_m.mkdir(parents=True, exist_ok=True)
        self.path_image_folder_color = Path(__file__).parents[0].joinpath(self.path_image_folder_m, "color")
        self.path_image_folder_color.mkdir(parents=True, exist_ok=True)
        self.path_image_folder_depth = Path(__file__).parents[0].joinpath(self.path_image_folder_m, "depth")
        self.path_image_folder_depth.mkdir(parents=True, exist_ok=True)

        self.path_image_folder_r = Path(__file__).parents[0].joinpath("stored_data_realsense", self.scene_name)
        self.path_image_folder_r.mkdir(parents=True, exist_ok=True)
        self.path_image_folder_color = Path(__file__).parents[0].joinpath(self.path_image_folder_r, "color")
        self.path_image_folder_color.mkdir(parents=True, exist_ok=True)
        self.path_image_folder_depth = Path(__file__).parents[0].joinpath(self.path_image_folder_r, "depth")
        self.path_image_folder_depth.mkdir(parents=True, exist_ok=True)

        #Create Marker
        self.marker_pub = rospy.Publisher("visualization_marker_array", MarkerArray, queue_size = 2)
        self.markerArray = MarkerArray()

        #Initialise Cameras
        self.mechmind = MechCapture()
        self.realsense = RealsenseCapture()

        #Add color to walls
        self.colors = dict()
        self.scene_pub = rospy.Publisher('planning_scene', moveit_msgs.msg.PlanningScene, queue_size=5)
    
    def create_next_folder(self):
        """
        Scans the specified directory, finds the last numbered folder,
        and creates a new folder with the next available number.
        """
        max_number = 0

        # Get a list of all folders in the directory
        folders = [folder for folder in os.listdir(self.dir) if os.path.isdir(os.path.join(self.dir, folder))]

        # Find the last numbered folder
        for folder in folders:
            if folder.startswith(f"{scene_type}_"):
                try:
                    folder_number = int(folder.split("_")[1])
                    max_number = max(max_number, folder_number)
                except (ValueError, IndexError):
                    continue
            else:
                folder_number = int("000")
                max_number = max(max_number, folder_number)
        # Create the next numbered folder
        next_number = max_number + 1
        # print(f"Created folder: {scene_type}_{next_number:03d}") 
        return f"{scene_type}_{next_number:03d}"

    def set_cartesian_path(self, scale=1):
        move_group = self.move_group
        waypoints = []
        #Move to default pose 
        wpose = move_group.get_current_pose().pose
        wpose.position.x = -0.14807260620066637
        wpose.position.y = 0.20937025277437452
        wpose.position.z = 0.6695901989593444
        wpose.orientation.x = -0.9995510768714108
        wpose.orientation.y = 0.009763006555211807
        wpose.orientation.z = -0.009329373517297389
        wpose.orientation.w = 0.02674492882736624
        waypoints.append(copy.deepcopy(wpose))

        #Move to calculated waypoints 
        waypoints.extend(self.move_spiral_ellipse())

        #Move to default pose
        wpose.position.x = -0.14807260620066637
        wpose.position.y = 0.20937025277437452
        wpose.position.z = 0.6695901989593444
        wpose.orientation.x = -0.9995510768714108
        wpose.orientation.y = 0.009763006555211807
        wpose.orientation.z = -0.009329373517297389
        wpose.orientation.w = 0.02674492882736624
        waypoints.append(copy.deepcopy(wpose))

        return waypoints

    def plan_path(self, waypoints):
        (plan, fraction) = self.move_group.compute_cartesian_path(waypoints, 0.01, 0.0  # waypoints to follow  # eef_step
                                                                  )  # jump_threshold

        return plan, fraction

    def display_trajectory(self, plan):
        robot = self.robot
        display_trajectory_publisher = self.display_trajectory_publisher
        display_trajectory = moveit_msgs.msg.DisplayTrajectory()
        display_trajectory.trajectory_start = robot.get_current_state()
        display_trajectory.trajectory.append(plan)
        display_trajectory_publisher.publish(display_trajectory)

    def execute_plan(self, plan):
        self.move_group.execute(plan, wait=True)

    def plan_and_execute(self):
        way = self.set_cartesian_path()
        counter = -2
        for i in way:
            new_way = []
            counter += 1            
            new_way.append(copy.deepcopy(i))
            plan, fraction = self.plan_path(new_way)
            self.display_trajectory(plan)
            self.execute_plan(plan)
            time.sleep(0.5)
            if counter == -1 or counter == num_waypoints:
                continue
            self.draw_marker_array(i, counter)
            print("\n" + self.scene_name + " | " + str(counter))
            self.realsense.capture_realsense_color(counter, self.path_image_folder_r)
            self.realsense.capture_realsense_depth(counter, self.path_image_folder_r)
            self.mechmind.capture_color_image(counter, self.path_image_folder_m)
            self.mechmind.capture_depth(counter, self.path_image_folder_m)
            
        self.mechmind.disconnect()
    
    def move_spiral_ellipse(self, spacing = 0.001):
        # Define parameters
        a = 0.0  # parameter of ellipse
        b = 0.01 # parameter of ellipse
        h = 0.1  # x-coordinate of center, 10cm from baselink
        k = 0.35 # y-coordinate of center, 35cm from baselink
        width = 0.5 # width of the ellipse
        length = 0.8 # length of the ellipse
        # num_waypoints = 256 # number of waypoints
        radius = 0.65 # distance from tool0 to center
        # Define theta range
        theta_start = 0
        theta_end = 12 * math.pi
        num_points = 3000 # if recursion error happens, decrease this number
        d_theta = (theta_end - theta_start) / (num_points - 1)
        theta_list = [theta_start + i * d_theta for i in range(num_points)]
        spiral_ellipse_waypoints = []
        wpose = Pose()
        for i in range(len(theta_list)):
            r = a + b * theta_list[i]
            x = r * math.cos(theta_list[i]) * length
            y = r * math.sin(theta_list[i]) * width
            z = math.sqrt(radius ** 2 - (10 * x - y) ** 2 / (1 + 100))
            if len(spiral_ellipse_waypoints) == 0:
                # first point
                new_x = x + h
                new_y = y + k
            else:
                distance = math.sqrt((x+h - new_x) ** 2 + (y+k - new_y) ** 2)
                if distance < spacing:
                    # skip this point
                    continue
            new_x = x + h
            new_y = y + k
            wpose.position.x = x + h
            wpose.position.y = y + k
            wpose.position.z = z
            item_x_coord = h
            item_y_coord = k
            if np.abs(wpose.position.x - item_x_coord) > 0.01:
                x_angle = np.arctan((wpose.position.x - item_x_coord)/wpose.position.z)
            else:
                x_angle = 0.0

            if np.abs(wpose.position.y - item_y_coord) > 0.01:
                y_angle = np.arctan((item_y_coord - wpose.position.y)/wpose.position.z)
            else:
                y_angle = 0.0
            trans = quaternion_from_euler(y_angle, x_angle, 0) # Pitch, Roll, Yaw

            # Quat values for face down
            # quat = [-0.5, 0.5, 0.5, 0.5]  # ur5 values
            quat = [-1, 0, 0, 0]            # ur5e values
            camera_quat = quaternion_multiply(trans, quat)
            wpose.orientation.x = camera_quat[0]
            wpose.orientation.y = camera_quat[1]
            wpose.orientation.z = camera_quat[2]
            wpose.orientation.w = camera_quat[3]
            spiral_ellipse_waypoints.append(copy.deepcopy(wpose))
            if len(spiral_ellipse_waypoints) > num_waypoints:
                # print(len(spiral_ellipse_waypoints))
                return self.move_spiral_ellipse(spacing = spacing + 0.0001)
        print("Number of waypoints: " + str(len(spiral_ellipse_waypoints)))
        return spiral_ellipse_waypoints

    def draw_marker_array(self, pose, counter):
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = rospy.Time.now()
        marker.type = 2 #arrow 0, cube 1, sphere 2 
        marker.id = 0

        if marker.type == 0:
        # Set the scale of the marker
            marker.type = marker.ARROW 
            marker.scale.x = 0.0005
            marker.scale.y = 0.0005
            marker.scale.z = 0.70
        elif marker.type == 1:
            marker.type = marker.CUBE
            marker.scale.x = 0.005
            marker.scale.y = 0.005
            marker.scale.z = 1.30
        else:
            marker.scale.x = 0.01
            marker.scale.y = 0.01
            marker.scale.z = 0.01

        # Set the color
        hue = counter / 256.0
        rgb = colorsys.hsv_to_rgb(hue, 1, 1)
        scaled_rgb = tuple(int(val * 255) for val in rgb)
        marker.color.r = scaled_rgb[0]/255.0
        marker.color.g = scaled_rgb[1]/255.0
        marker.color.b = scaled_rgb[2]/255.0
        marker.color.a = 1
        # Calculate pose of the camera
        tool0_pose = self.get_transform_mat_from_pose([pose.position.x, pose.position.y, pose.position.z, pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        tool0_to_cam = self.get_transform_mat_from_pose([0.02598823968408548, -0.13109019051557816, 0.05607019307819193, 0.0008547100104746553, 0.056641822866922036, 0.00019581856172904235, -0.9983941781822454])
        cam_pose = self.get_pose_from_transform_mat(np.matmul(tool0_pose, tool0_to_cam))
        # Set the pose of the marker
        marker.pose.position.x = cam_pose[0]
        marker.pose.position.y = cam_pose[1]
        marker.pose.position.z = cam_pose[2] 
        marker.pose.orientation.x = cam_pose[3]
        marker.pose.orientation.y = cam_pose[4]
        marker.pose.orientation.z = cam_pose[5]
        marker.pose.orientation.w = cam_pose[6]
        self.markerArray.markers.append(marker)
        for m in self.markerArray.markers:
            m.id = counter

        self.marker_pub.publish(self.markerArray)

    def add_box(self):
        scene = self.scene
        if len(self.scene.get_known_object_names()) > 0:
            print("Walls already added")
            return True
        print("Adding Walls")
        rospy.sleep(3)
        floor_pose = PoseStamped()
        floor_pose.header.frame_id = "base"
        floor_pose.pose.orientation.w = 1.0
        floor_pose.pose.position.x = 0
        floor_pose.pose.position.y = 0
        floor_pose.pose.position.z = -0.03
        box_name = "floor"
        scene.add_box(box_name, floor_pose, size=(1, 1, 0.05))
        self.setColor(box_name, 0.8, 0.8, 0.8, 0.1)
        self.sendColors()

        wall1_pose = PoseStamped()
        wall1_pose.header.frame_id = "base"
        wall1_pose.pose.orientation.w = 1.0
        wall1_pose.pose.position.x = 0
        wall1_pose.pose.position.y = 0.75
        wall1_pose.pose.position.z = 0.5
        box_name = "back_wall"
        scene.add_box(box_name, wall1_pose, size=(1, 0.05, 1))
        self.setColor(box_name, 0.8, 0.8, 0.8, 0.1)
        self.sendColors()

        wall2_pose = PoseStamped()
        wall2_pose.header.frame_id = "base"
        wall2_pose.pose.orientation.w = 1.0
        wall2_pose.pose.position.x = -0.75
        wall2_pose.pose.position.y = 0
        wall2_pose.pose.position.z = 0.5
        box_name = "right_wall"
        scene.add_box(box_name, wall2_pose, size=(0.05, 1, 1))
        self.setColor(box_name, 0.8, 0.8, 0.8, 0.1)
        self.sendColors()

        wall3_pose = PoseStamped()
        wall3_pose.header.frame_id = "base"
        wall3_pose.pose.orientation.w = 1.0
        wall3_pose.pose.position.x = 0
        wall3_pose.pose.position.y = -0.30 
        wall3_pose.pose.position.z = 0.22
        box_name = "obstacles"
        scene.add_box(box_name, wall3_pose, size=(1, 0.01, 0.50))
        self.setColor(box_name, 0.8, 0.8, 0.8, 0.1)
        self.sendColors()

        ceiling_pose = PoseStamped()
        ceiling_pose.header.frame_id = "base"
        ceiling_pose.pose.orientation.w = 1.0
        ceiling_pose.pose.position.x = 0
        ceiling_pose.pose.position.y = 0 
        ceiling_pose.pose.position.z = 0.9
        box_name = "ceiling"
        scene.add_box(box_name, ceiling_pose, size=(1, 1, 0.05))
        self.setColor(box_name, 0.8, 0.8, 0.8, 0.1)
        self.sendColors()
        self.box_name = box_name
        rospy.sleep(3)

    def setColor(self,name,r,g,b,a=0.9):
        color = moveit_msgs.msg.ObjectColor()
        color.id=name
        color.color.r=r
        color.color.g=g
        color.color.b=b
        color.color.a=a
        self.colors[name]=color
		    
    def sendColors(self):
        p = moveit_msgs.msg.PlanningScene()
        p.is_diff=True
        for color in self.colors.values():
            p.object_colors.append(color)
        self.scene_pub.publish(p)


    def get_transform_mat_from_pose(self, transform):
        """
        get transform matrix from ROS pose
        transform matrix: [4*4]
        pose: [7], (translation [3], orientation [4])
        """
        transform_mat = np.zeros(shape=(4,4))  
        transform_mat[0:3,0:3] = R.from_quat([transform[3], transform[4], transform[5], transform[6]]).as_matrix()
        transform_mat[0][3] = transform[0]
        transform_mat[1][3] = transform[1]
        transform_mat[2][3] = transform[2]
        transform_mat[3][3] = 1

        return transform_mat

    def get_pose_from_transform_mat(self, transform_mat):
        """
        get ROS pose from transform matrix
        pose: [7], (translation [3], orientation [4])
        transform matrix: [4*4]
        """
        quat = R.from_matrix(transform_mat[0:3,0:3]).as_quat()
        return [transform_mat[0,3], transform_mat[1,3], transform_mat[2,3], quat[0], quat[1], quat[2], quat[3]]

def main():
    try:
        bot = MoveGroupPythonInterface()
        bot.add_box()
        bot.plan_and_execute()
        print("time took:" + str(time.time() - start_time))
    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        return

if __name__ == "__main__":
    start_time = time.time()
    rospy.init_node("JA_UR_Program")
    num_waypoints = 256
    scene = input("Enter scene type: 1 for Simple, 2 for Mixed, 3 for Complex: ")
    # scene = '3' 
    if scene == '1':
        scene_type = 'A'
    elif scene == '2':
        scene_type = 'B'
    elif scene == '3':
        scene_type = 'C'
    else:
        print('Invalid scene type')
        exit()
    main()
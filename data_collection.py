from __future__ import print_function
from six.moves import input
#need to import MechMind first
from capture_mechmind import MechCapture

import sys
import os
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import time
import tf

from moveit_commander.conversions import pose_to_list
from tf.transformations import *

from geometry_msgs.msg import PoseStamped, Pose
import numpy as np
from scipy.spatial.transform import Rotation as R

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

from datetime import datetime
from pathlib import Path
import colorsys

start_time = time.time()
rospy.init_node("JA_UR_Program")

import capture_realsense as realsense

mechmind = MechCapture()

marker_pub = rospy.Publisher("visualization_marker_array", MarkerArray, queue_size = 2)
markerArray = MarkerArray()

class MoveGroupPythonInterface(object):

    def __init__(self):
        super(MoveGroupPythonInterface, self).__init__()
        moveit_commander.roscpp_initialize(sys.argv)
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

        #rospy.init_node("move_group_python_interface_tutorial", anonymous=True)
        robot = moveit_commander.RobotCommander()
        scene = moveit_commander.PlanningSceneInterface()
        group_name = "manipulator"
        move_group = moveit_commander.MoveGroupCommander(group_name)
        display_trajectory_publisher = rospy.Publisher(
            "/move_group/display_planned_path",
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=20,
        )
        planning_frame = move_group.get_planning_frame()
        eef_link = move_group.get_end_effector_link()
        group_names = robot.get_group_names()

        self.box_name = " "
        self.robot = robot
        self.scene = scene
        self.move_group = move_group
        self.display_trajectory_publisher = display_trajectory_publisher
        self.planning_frame = planning_frame
        self.eef_link = eef_link
        self.group_names = group_names
        #Add color to wall
        self.colors = dict()
        self.scene_pub = rospy.Publisher('planning_scene', moveit_msgs.msg.PlanningScene, queue_size=5)
        mechmind.find_camera_list()
        mechmind.connect_device_info()
    
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
        print(f"Created folder: {scene_type}_{next_number:03d}") 
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

        #Taking samples
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
    
    def plan_cartesian_path(self, pose_object, scale=1):
        # plan a Cartesian path directly by specifying a list of waypoints for the end-effector to go through
        print("============ Planning cartesian path ")
        waypoints = []

        if isinstance(pose_object, list):
            wpose = self.move_group.get_current_pose().pose
            wpose.position.x = float(pose_object[0])
            wpose.position.y = float(pose_object[1])
            wpose.position.z = float(pose_object[2])
            wpose.orientation.x = float(pose_object[3])
            wpose.orientation.y = float(pose_object[4])
            wpose.orientation.z = float(pose_object[5])
            wpose.orientation.w = float(pose_object[6])
            waypoints.append(copy.deepcopy(wpose))

        elif isinstance(pose_object, geometry_msgs.msg._Pose.Pose):
            waypoints.append(copy.deepcopy(pose_object))

        (plan, fraction) = self.move_group.compute_cartesian_path(waypoints, 0.01, 0.0)
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
        ct = -2
        for i in way:
            new_way = []
            ct+=1            
            new_way.append(copy.deepcopy(i))
            plan, fraction = self.plan_path(new_way)
            self.display_trajectory(plan)
            self.execute_plan(plan)
            time.sleep(0.5)
            if ct == -1 or ct == 256:
                continue
            self.draw_sphere_array(i, ct)
            print("\n" + self.scene_name + " | " + str(ct))
            realsense.capture_realsense_color(ct, self.path_image_folder_r)
            realsense.capture_realsense_depth(ct, self.path_image_folder_r)
            mechmind.capture_color_image(ct, self.path_image_folder_m)
            mechmind.capture_depth(ct, self.path_image_folder_m)
            
        mechmind.disconnect()
    
    def move_spiral_ellipse(self):
        # Define parameters
        a = 0.0  # parameter of ellipse
        b = 0.01 # parameter of ellipse
        h = 0.1  # x-coordinate of center
        k = 0.35 # y-coordinate of center
        width = 0.5 # width of the ellipse
        length = 0.8 # length of the ellipse
        num_waypoints = 256 # number of waypoints
        radius = 0.65 # distance from tool0 to center
        # Define theta range
        theta_start = 0
        theta_end = 12 * math.pi
        num_points = 1000
        d_theta = (theta_end - theta_start) / (num_points - 1)
        theta_list = [theta_start + i * d_theta for i in range(num_points)]
        spiral_ellipse_waypoints = []
        wpose = Pose()
        total_length = 4.747206936602081 # length of the spiral_ellipse
        spacing = total_length / num_waypoints # spacing between waypoints
        current_spacing = 0
        point_id = 0
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
                current_spacing += distance
                if current_spacing < spacing:
                    # skip this point
                    continue
            point_id += 1
            new_x = x + h
            new_y = y + k
            current_spacing = 0
            wpose.position.x = x + h
            wpose.position.y = y + k
            wpose.position.z = z
            item_x_coord = h
            item_y_coord = k
            curr_pose = self.move_group.get_current_pose().pose
            curr_pose.position.z = wpose.position.z # Make sure z is at the same height
            if np.abs(wpose.position.x - item_x_coord) > 0.01:
                x_angle = np.arctan((wpose.position.x - item_x_coord)/curr_pose.position.z)
            else:
                x_angle = 0.0

            if np.abs(wpose.position.y - item_y_coord) > 0.01:
                y_angle = np.arctan((item_y_coord - wpose.position.y)/curr_pose.position.z)
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
            # Delete the even point if the number of waypoints is greater than the number of desired points
            if len(spiral_ellipse_waypoints) > num_waypoints:
                spiral_ellipse_waypoints.pop(int(point_id - num_waypoints))

        return spiral_ellipse_waypoints

    def draw_sphere_array(self, pose, counter):
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = rospy.Time.now()
        marker.type = 1 #arrow 0, cube 1, sphere 2 
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
        # print(scaled_rgb)
        marker.color.r = scaled_rgb[0]/255.0
        marker.color.g = scaled_rgb[1]/255.0
        marker.color.b = scaled_rgb[2]/255.0
        marker.color.a = 0.5
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
        markerArray.markers.append(marker)
        for m in markerArray.markers:
            m.id = counter

        marker_pub.publish(markerArray)

    def add_box(self):

        scene = self.scene
        rospy.sleep(3)
        floor_pose = geometry_msgs.msg.PoseStamped()
        floor_pose.header.frame_id = "base"
        floor_pose.pose.orientation.w = 1.0
        floor_pose.pose.position.x = 0
        floor_pose.pose.position.y = 0
        floor_pose.pose.position.z = -0.03
        box_name = "floor"
        scene.add_box(box_name, floor_pose, size=(1, 1, 0.05))
        self.setColor(box_name, 0.8, 0.8, 0.8, 0.1)
        self.sendColors()

        wall1_pose = geometry_msgs.msg.PoseStamped()
        wall1_pose.header.frame_id = "base"
        wall1_pose.pose.orientation.w = 1.0
        wall1_pose.pose.position.x = 0
        wall1_pose.pose.position.y = 0.75
        wall1_pose.pose.position.z = 0.5
        box_name = "back_wall"
        scene.add_box(box_name, wall1_pose, size=(1, 0.05, 1))
        self.setColor(box_name, 0.8, 0.8, 0.8, 0.1)
        self.sendColors()

        wall2_pose = geometry_msgs.msg.PoseStamped()
        wall2_pose.header.frame_id = "base"
        wall2_pose.pose.orientation.w = 1.0
        wall2_pose.pose.position.x = -0.75
        wall2_pose.pose.position.y = 0
        wall2_pose.pose.position.z = 0.5
        box_name = "right_wall"
        scene.add_box(box_name, wall2_pose, size=(0.05, 1, 1))
        self.setColor(box_name, 0.8, 0.8, 0.8, 0.1)
        self.sendColors()

        wall3_pose = geometry_msgs.msg.PoseStamped()
        wall3_pose.header.frame_id = "base"
        wall3_pose.pose.orientation.w = 1.0
        wall3_pose.pose.position.x = 0
        wall3_pose.pose.position.y = -0.25 
        wall3_pose.pose.position.z = 0.25
        box_name = "obstacles"
        scene.add_box(box_name, wall3_pose, size=(1, 0.01, 0.50))
        self.setColor(box_name, 0.8, 0.8, 0.8, 0.1)
        self.sendColors()

        ceiling_pose = geometry_msgs.msg.PoseStamped()
        ceiling_pose.header.frame_id = "base"
        ceiling_pose.pose.orientation.w = 1.0
        ceiling_pose.pose.position.x = 0
        ceiling_pose.pose.position.y = 0 
        ceiling_pose.pose.position.z = 0.9
        print("adding box")
        box_name = "ceiling"
        scene.add_box(box_name, ceiling_pose, size=(1, 1, 0.05))
        self.setColor(box_name, 0.8, 0.8, 0.8, 0.1)
        self.sendColors()
        self.box_name = box_name
        rospy.sleep(5)

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
    # scene = input("Enter scene type: 1 for Simple, 2 for Mixed, 3 for Complicated: ")
    scene = '2' 
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
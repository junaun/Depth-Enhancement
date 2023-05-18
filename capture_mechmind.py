from MechEye import Device
import rospy 
import cv2
import sys
sys.path.append('/home/arm-orin-01/ja_ws/src/ja_script/src')

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from pathlib import Path
from datetime import datetime
import numpy as np
import tf
import geometry_msgs.msg
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import tf2_ros

br = CvBridge()

#rospy.init_node("mechmind_tf_listener")


def show_error(status):
    if status.ok():
        return
    print("Error Code : {}".format(status.code()),
          ",Error Description: {}".format(status.description()))


def print_device_info(num, info):
    print(" Mech-Eye device index: {}\n".format(str(num)),
          "Camera Model Name: {}\n".format(info.model()),
          "Camera ID: {}\n".format(info.id()),
          "Camera IP: {}\n".format(info.ip()),
          "Hardware Version: {}\n".format(info.hardware_version()),
          "Firmware Version: {}\n".format(info.firmware_version()),
          "...............................................")

'''def get_mechmind_tf(path_npy_folder, counter):
    listener = tf.TransformListener()
    listener.waitForTransform('/mechmind_camera/color_map', '/base_link', rospy.Time(), rospy.Duration(4.0))
    (trans, rot) = listener.lookupTransform('/mechmind_camera/color_map', '/base_link', rospy.Time(0))
    pose_msg = geometry_msgs.msg.Pose()
    pose_msg.position.x = trans[0]
    pose_msg.position.y = trans[1]
    pose_msg.position.z = trans[2]
    pose_msg.orientation.x = rot[0]
    pose_msg.orientation.y = rot[1]
    pose_msg.orientation.z = rot[2]
    pose_msg.orientation.w = rot[3]
    new_matrix = get_transform_mat_from_pose(trans,rot)
    array.append(new_matrix)
    np.array(array)
    #print(array)
    #print(new_matrix)
    npy_filename = str(path_npy_folder.joinpath(f"tf_mechmind_to_baselink"))
    np.save(npy_filename, array)
    #print("Saved tf_realsense_to_baselink.npy")

def get_transform_mat_from_pose(transform, rotation):
    """
    get transform matrix from ROS pose
    transform matrix: [4*4]
    pose: [7], (translation [3], orientation [4])
    """
    transform_mat = np.zeros(shape=(4,4))  
    transform_mat[0:3,0:3] = R.from_quat([rotation[0], rotation[1], rotation[2], rotation[3]]).as_matrix()
    transform_mat[0][3] = transform[0]
    transform_mat[1][3] = transform[1]
    transform_mat[2][3] = transform[2]
    transform_mat[3][3] = 1

    return transform_mat

def capture_mechmind_color(counter):
    this_img_filename = str(path_image_folder_color.joinpath(f"{counter:02}.png")) # the :02 is to represent number of digits

    # Saving colour image into file
    cam = CapturePointCloud()
    img_color = cam.capture_color_image()
    cv2.imwrite(this_img_filename, img_color)
    get_mechmind_tf(path_image_folder, counter)

    print(f"Saved color{counter}.png")

def capture_mechmind_depth(counter):
    this_img_filename = str(path_image_folder_depth.joinpath(f"{counter:02}.png")) # the :02 is to represent number of digits

    cam = CapturePointCloud()
    img_depth = cam.capture_depth()
    cv2.imwrite(this_img_filename, img_depth)
    get_mechmind_tf(path_image_folder, counter)

    print(f"Saved depth{counter}.png")'''

class MechCapture(object):
    def __init__(self):
        self.device = Device()
        self.device.set_scan_2d_exposure_mode("Auto")
        self.find_camera_list()
        self.connect_device_info()
        self.array = []
        self.now = datetime.now().strftime("%d%m%y_%H%MH")
        self.path_image_folder = Path(__file__).parents[0].joinpath("stored_data_mechmind", self.now)
        # self.path_image_folder.mkdir(parents=True, exist_ok=True)
        self.path_image_folder_color = Path(__file__).parents[0].joinpath(self.path_image_folder, "color")
        # self.path_image_folder_color.mkdir(parents=True, exist_ok=True)
        self.path_image_folder_depth = Path(__file__).parents[0].joinpath(self.path_image_folder, "depth")
        # self.path_image_folder_depth.mkdir(parents=True, exist_ok=True)
        #self.path_image_folder_point = Path(__file__).parents[0].joinpath(self.path_image_folder, "Point_Cloud")
        #self.path_image_folder_point.mkdir(parents=True, exist_ok=True)
        self.device.set_scan_2d_exposure_mode("Auto")
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        #self.device.set_depth_range(0, 4000)
        #print(self.device.get_depth_range().lower())
        #print(self.device.get_depth_range().upper())
        

    def find_camera_list(self):
        print("Find Mech-Eye devices...")
        self.device_list = self.device.get_device_list()
        if len(self.device_list) == 0:
            print("No Mech-Eye device found.")
            quit()
        for i, info in enumerate(self.device_list):
            print_device_info(i, info)

    def connect_device_info(self):
        status = self.device.connect(self.device_list[0])
        if not status.ok():
            show_error(status)
            quit()
        print("Connected to the Mech-Eye device successfully.")

    def capture_point_cloud(self, counter):
        self.device.set_scan_2d_exposure_mode("Auto")
        #self.device.set_scan_2d_exposure_time(60.0)
        color = self.device.capture_color()
        color_data = color.data()
        point_xyz = self.device.capture_point_xyz()
        point_xyz_data = point_xyz.data()

        point_cloud_xyz = o3d.geometry.PointCloud()
        points_xyz = np.zeros(
            (point_xyz.width() * point_xyz.height(), 3), dtype=np.float64)

        pos = 0
        for dd in np.nditer(point_xyz_data):
            points_xyz[int(pos / 3)][int(pos % 3)] = 0.001 * dd
            pos = pos + 1

        point_cloud_xyz.points = o3d.utility.Vector3dVector(points_xyz)
        #o3d.visualization.draw_geometries([point_cloud_xyz])
        #o3d.io.write_point_cloud("PointCloudXYZ.ply", point_cloud_xyz)
        #print("Point cloud saved to path PointCloudXYZ.ply")

        point_cloud_xyz_rgb = o3d.geometry.PointCloud()
        point_cloud_xyz_rgb.points = o3d.utility.Vector3dVector(points_xyz)
        points_rgb = np.zeros(
            (point_xyz.width() * point_xyz.height(), 3), dtype=np.float64)

        pos = 0
        for dd in np.nditer(color_data):
            points_rgb[int(pos / 3)][int(2 - (pos % 3))] = dd / 255
            pos = pos + 1

        point_cloud_xyz_rgb.colors = o3d.utility.Vector3dVector(points_rgb)
        this_img_filename = str(self.path_image_folder_point.joinpath(f"{counter:02}.ply")) # the :02 is to represent number of digits
        #o3d.visualization.draw_geometries([point_cloud_xyz_rgb])
        o3d.io.write_point_cloud(this_img_filename, point_cloud_xyz_rgb)
        print(f"Saved mechmind_pc{counter}.ply")
        # print("Color point cloud saved to path PointCloudXYZRGB.ply")
        #return point_cloud_xyz_rgb

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
        #depth_map = cv2.cvtColor(depth_map.data(), cv2.COLOR_BGR2GRAY)
        #depth_map = cv2.convertScaleAbs(depth_map.data(), alpha=(255.0/np.max(depth_map.data())))
        '''print(depth_map.min())
        print(depth_map.max())
        print(depth_map)'''
        #cv2.imwrite(this_img_filename, cv2.equalizeHist(depth_map.data()).astype(np.uint16))
        cv2.imwrite(this_img_filename, depth_map.data().astype(np.uint16))
        #.astype(np.uint16)
        #print(depth_map.shape)
        #print(np.nonzero(depth_map))
        print(f"Saved mechmind_depth{counter}.png")

    def get_mechmind_tf(self, path_npy_folder, counter):
        #listener = tf.TransformListener()
        try:
            #listener.waitForTransform('/mechmind_camera/color_map', '/base_link', rospy.Time(), rospy.Duration(4.0))
            #(trans, rot) = listener.lookupTransform('/mechmind_camera/color_map', '/base_link', rospy.Time())
            #(trans, rot) = self.tfBuffer.lookup_transform("mechmind_camera/color_map","base_link",  rospy.Time())
            trans = self.tfBuffer.lookup_transform("base_link","mechmind_camera/color_map",  rospy.Time(0))

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as ex:
            rospy.logwarn(str(ex))
            rospy.Rate(10.0).sleep()
            return
        #listener.waitForTransform('/mechmind_camera/color_map', '/base_link', rospy.Time(), rospy.Duration(4.0))
        #t = listener.getLatestCommonTime("/mechmind_camera/color_map", "/base_link")
        #(trans, rot) = listener.lookupTransform('/mechmind_camera/color_map', '/base_link', t)
        #(trans, rot) = self.tfbuffer.lookup_transform('mechmind_camera/color_map', 'base_link', t)
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
        #print(array)
        #print(new_matrix)
        npy_filename = str(path_npy_folder.joinpath(f"tf_mechmind_to_baselink"))
        np.save(npy_filename, self.array)
        #print("Saved tf_realsense_to_baselink.npy")

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

    def main(self):
        self.find_camera_list()
        self.connect_device_info()
        self.capture_point_cloud()


def main():
    rospy.init_node("mechmind_tf_listener")
    counter = 0
    cam = MechCapture()
    while True:
        value = input("'n' to capture image, 'q' to quit: ")
        if value == 'n':
            cam.capture_color_image(counter)
            cam.capture_depth(counter)
            counter+=1

if __name__=="__main__":
    #main()
    rospy.init_node("mechmind_tf_listener")
    cam = MechCapture()
    #cam.get_mechmind_tf(cam.path_image_folder, 0)
    cam.capture_point_cloud(00)
    cam.disconnect()
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
import tf2_ros

br = CvBridge()


array = []
now = datetime.now().strftime("%d%m%y_%H%MH")
path_image_folder = Path(__file__).parents[0].joinpath("stored_data_realsense", now)
# path_image_folder.mkdir(parents=True, exist_ok=True)
path_image_folder_color = Path(__file__).parents[0].joinpath(path_image_folder, "color")
# path_image_folder_color.mkdir(parents=True, exist_ok=True)
path_image_folder_depth = Path(__file__).parents[0].joinpath(path_image_folder, "depth")
# path_image_folder_depth.mkdir(parents=True, exist_ok=True)
tfBuffer = tf2_ros.Buffer()
listener = tf2_ros.TransformListener(tfBuffer)

def get_realsense_tf(path_npy_folder, counter):
    #listener = tf.TransformListener()
    #listener.waitForTransform('/camera_color_optical_frame', '/base_link', rospy.Time(), rospy.Duration(4.0))
    #t = listener.getLatestCommonTime("/camera_color_optical_frame", "/base_link")
    #(trans, rot) = listener.lookupTransform('/camera_color_optical_frame', '/base_link', t)
    #(trans, rot) = tfbuffer.lookup_transform('camera_color_optical_frame', 'base_link', rospy.Time())
    try:
        #listener.waitForTransform('/mechmind_camera/color_map', '/base_link', rospy.Time(), rospy.Duration(4.0))
        #(trans, rot) = listener.lookupTransform('/mechmind_camera/color_map', '/base_link', rospy.Time())
        #(trans, rot) = self.tfBuffer.lookup_transform("mechmind_camera/color_map","base_link",  rospy.Time())
        trans = tfBuffer.lookup_transform("base_link",  "realsense_camera", rospy.Time(0))

        pose_msg = []
        pose_msg.append(trans.transform.translation.x)
        pose_msg.append(trans.transform.translation.y)
        pose_msg.append(trans.transform.translation.z)
        pose_msg.append(trans.transform.rotation.x)
        pose_msg.append(trans.transform.rotation.y)
        pose_msg.append(trans.transform.rotation.z)
        pose_msg.append(trans.transform.rotation.w)
        new_matrix = get_transform_mat_from_pose(pose_msg)
        array.append(new_matrix)
        np.array(array)
        #print(array)
        #print(new_matrix)
        npy_filename = str(path_npy_folder.joinpath(f"tf_realsense_to_baselink"))
        np.save(npy_filename, array)
        #print("Saved tf_realsense_to_baselink.npy")
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as ex:
        rospy.logwarn(str(ex))
        #rospy.Rate(10.0).sleep()
        pass


def get_transform_mat_from_pose(transform):
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

def capture_realsense_color(counter, path):
    this_img_filename = str(path.joinpath(f"color/{counter:03}.png")) # the :02 is to represent number of digits

    # Saving colour image into file
    data = rospy.wait_for_message("/camera/color/image_raw", Image)
    img_color = br.imgmsg_to_cv2(data, desired_encoding='bgr8') 
    cv2.imwrite(this_img_filename, img_color)
    get_realsense_tf(path, counter)

    print(f"Saved realsense_color{counter}.png")

def capture_realsense_depth(counter, path):
    this_img_filename = str(path.joinpath(f"depth/{counter:03}.png")) # the :02 is to represent number of digits

    # Saving colour image into file
    #data = rospy.wait_for_message("/camera/depth/image_rect_raw", Image)
    data = rospy.wait_for_message("/camera/aligned_depth_to_color/image_raw", Image)
    img_depth = br.imgmsg_to_cv2(data, desired_encoding='passthrough')
    #img_depth = cv2.convertScaleAbs(img_depth, alpha=(255.0/np.max(img_depth)))
 
    cv2.imwrite(this_img_filename, img_depth.astype(np.uint16))
    #get_realsense_tf(path_image_folder, counter)

    print(f"Saved realsense_depth{counter}.png")

def cam_tf_trans():
    br = tf2_ros.StaticTransformBroadcaster()
    static_transformStamped = geometry_msgs.msg.TransformStamped()
    static_transformStamped.header.frame_id = "tool0"
    static_transformStamped.child_frame_id = "camera_link"
    
    # Manually copy xyzquat values from calibration file if there's any changes
    pose_tool2camlink = [-0.11829361021809054, 0.03136173945462838, 0.027655245112394273, -0.0025304382943000074, -0.0026573084677070625, -0.7045908171540352, 0.7096043376248169]
    mat_tool2camlink = get_transform_mat_from_pose(pose_tool2camlink)

    #camlink2cambase_pose = lookup_transform('camera_color_optical_frame', 'camera_link')
    #camlink2cambase_pose = lookup_transform('camera_aligned_depth_to_camera_frame', 'camera_link')
    camlink2cambase_pose = lookup_transform('tool0', 'base_link')
    mat_camlink2cambase = get_transform_mat_from_pose(camlink2cambase_pose)
    mat_tool2cambase = np.matmul(mat_tool2camlink, mat_camlink2cambase)
    tool2cambase_pose = get_pose_from_transform_mat(mat_tool2cambase)

    static_transformStamped.transform.translation.x = tool2cambase_pose[0]
    static_transformStamped.transform.translation.y = tool2cambase_pose[1]
    static_transformStamped.transform.translation.z = tool2cambase_pose[2]
    static_transformStamped.transform.rotation.x = tool2cambase_pose[3]
    static_transformStamped.transform.rotation.y = tool2cambase_pose[4]
    static_transformStamped.transform.rotation.z = tool2cambase_pose[5]
    static_transformStamped.transform.rotation.w = tool2cambase_pose[6]

    # Time stamp should always be done last to be as accurate as possible
    static_transformStamped.header.stamp = rospy.Time.now()
    #br.sendTransform(static_transformStamped)
    print(static_transformStamped.transform)

def lookup_transform(from_here, to_here):
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
    

if __name__=="__main__":
    rospy.init_node("realsense_tf_listener")
    counter = 0
    while True:
        #value = input("'n' to capture image, 'q' to quit: ")
        #if value == 'n':
            #capture_realsense_color(counter)
            #capture_realsense_depth(counter)
            #cam_tf_trans()
            tfBuffer = tf2_ros.Buffer()
            listener = tf2_ros.TransformListener(tfBuffer)
            try:

                camlink2cambase = tfBuffer.lookup_transform('tool0', 'mechmind_camera/color_map', rospy.Time(0))
                camlink2cambase_pose = []
                camlink2cambase_pose.append(camlink2cambase.transform.translation.x)
                camlink2cambase_pose.append(camlink2cambase.transform.translation.y)
                camlink2cambase_pose.append(camlink2cambase.transform.translation.z)
                camlink2cambase_pose.append(camlink2cambase.transform.rotation.x)
                camlink2cambase_pose.append(camlink2cambase.transform.rotation.y)
                camlink2cambase_pose.append(camlink2cambase.transform.rotation.z)
                camlink2cambase_pose.append(camlink2cambase.transform.rotation.w)
                print( camlink2cambase_pose[0])

            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException)as ex:
                rospy.logwarn(str(ex))
                rospy.Rate(10).sleep()
                pass
#!/usr/bin/env python3.8

import open3d as o3d
import numpy as np
from time import time
from pathlib import Path
import copy
from multiprocessing import Process, cpu_count
import os
from matplotlib import pyplot as plt
import cv2
from probreg import cpd, filterreg, features, l2dist_regs, gmmtree, callbacks
import shutil

# ----------------------------Scan & Make Directory---------------------------------
def scan_dir():
    raw_folder_path = Path(__file__).parents[0].joinpath("stored_data_mechmind")
    raw_folders = []
    processed_folder_path = Path(__file__).parents[0].joinpath("combined_data")
    processed_folders = []

    for entry in os.scandir(processed_folder_path): 
        if entry.is_dir():
            processed_folders.append(entry.name)

    for entry in os.scandir(raw_folder_path):
        if entry.is_dir():
            folder_name = entry.name
            if folder_name in processed_folders:
                continue
            else:
                raw_folders.append(folder_name)
    print(f"Raw_folders : {raw_folders}, {len(raw_folders)} ")
    # print(f"Processed_folders : {processed_folders}")
    return raw_folders

def make_dir(folder_name):
    global path_image_folder
    path_image_folder = Path(__file__).parents[0].joinpath("combined_data", folder_name)
    path_image_folder.mkdir(parents=True, exist_ok=True)
    global path_image_folder_color
    path_image_folder_color = Path(__file__).parents[0].joinpath(path_image_folder, "color")
    path_image_folder_color.mkdir(parents=True, exist_ok=True)
    global path_image_folder_depth
    path_image_folder_depth = Path(__file__).parents[0].joinpath(path_image_folder, "depth")
    path_image_folder_depth.mkdir(parents=True, exist_ok=True)
    path_image_folder_edge = Path(__file__).parents[0].joinpath(path_image_folder, "edge")
    path_image_folder_edge.mkdir(parents=True, exist_ok=True)

def delete_folder(folder_name):
    print(f"Deleting {folder_name}")
    shutil.rmtree(f"combined_data/{folder_name}")

#----------------------------------------MechMind------------------------------------------
def process_mechmind_image(num, folder, mech_poses, crop):
    color = o3d.io.read_image(f"stored_data_mechmind/{folder}/color/{num:03}.png")
    depth = o3d.io.read_image(f"stored_data_mechmind/{folder}/depth/{num:03}.png")
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, convert_rgb_to_intensity=False)
    if crop:    
        pcd1 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, mech_intrinsic_value, np.linalg.inv(mech_poses[num])).crop(box)
    else:
        pcd1 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, mech_intrinsic_value, np.linalg.inv(mech_poses[num]))
    return pcd1

# ------------------------------------------RealSense------------------------------------------
def process_realsense_image(num, folder, real_poses, crop):
    color = o3d.io.read_image(f"stored_data_realsense/{folder}/color/{num:03}.png")
    o3d.io.write_image(f"{path_image_folder_color}/{num:03}_real.png", color)
    depth = o3d.io.read_image(f"stored_data_realsense/{folder}/depth/{num:03}.png")
    o3d.io.write_image(f"{path_image_folder_depth}/{num:03}_real.png", depth) 
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, convert_rgb_to_intensity=False)
    if crop:
        pcd2 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, real_intrinsic_value, np.linalg.inv(real_poses[num])).crop(box)
    else:
        pcd2 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, real_intrinsic_value, np.linalg.inv(real_poses[num]))
    return pcd2

# ------------------------------------------RANSAC + ICP--------------------------------------------
def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    # source_temp.paint_uniform_color([1, 0.706, 0])
    # target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def preprocess_point_cloud(pcd, voxel_size):
    # print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def prepare_dataset(voxel_size, source, target):
    #print(":: Load two point clouds and disturb initial pose.")

    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=300))
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=300))
    trans_init = np.identity(4)
    source.transform(trans_init)
    #draw_registration_result(source, target, np.identity(4))
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    # print(":: RANSAC registration on downsampled point clouds.")
    # print("   Since the downsampling voxel size is %.3f," % voxel_size)
    # print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def refine_registration(source, target, voxel_size, result_ransac):
    distance_threshold = voxel_size * 1.5  
    # print(":: Point-to-plane ICP registration is applied on original point")
    # print("   clouds to refine the alignment. This time we use a strict")
    # print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    evaluation = o3d.pipelines.registration.evaluate_registration(source, target, distance_threshold, result.transformation)
    return result, evaluation

# ------------------------------------------Project PointCloud to Depth/Color---------------------------------------
def point_cloud_to_depth_image(pcd, intrinsic, i, name):
    depth_image = np.zeros((intrinsic.height, intrinsic.width))
    print(f"processing Depth {i}")
    for point in pcd.points:
        z = point[2]*1000
        u = (point[0]*1000*intrinsic.intrinsic_matrix[0][0])/z #fx
        v = (point[1]*1000*intrinsic.intrinsic_matrix[1][1])/z #fy
        u = int(u + intrinsic.intrinsic_matrix[0][2])          #cx
        v = int(v + intrinsic.intrinsic_matrix[1][2])          #cy
        if 0 <= u < intrinsic.width and 0 <= v < intrinsic.height:
            depth_image[v, u] = z
    o3d.io.write_image(f"{path_image_folder_depth}/{i:03}_{name}.png",o3d.geometry.Image(depth_image.astype(np.uint16)))

def point_cloud_to_rgb_image(pcd, intrinsic, i, name):
    rgb_image = np.zeros((intrinsic.height, intrinsic.width, 3), dtype=np.uint8)
    print(f"processing RGB {i}")
    for id, point in enumerate(pcd.points):
        z = point[2]*1000
        u = (point[0]*1000*intrinsic.intrinsic_matrix[0][0])/z #fx
        v = (point[1]*1000*intrinsic.intrinsic_matrix[1][1])/z #fy
        u = int(u + intrinsic.intrinsic_matrix[0][2])          #cx
        v = int(v + intrinsic.intrinsic_matrix[1][2])          #cy
        color = pcd.colors[id]
        colors = [int(c * 255) for c in color]
        #print(u,v, colors)
        if 0 <= u < intrinsic.width and 0 <= v < intrinsic.height:
            rgb_image[v, u] = colors
    o3d.io.write_image(f"{path_image_folder_color}/{i:03}_{name}.png", o3d.geometry.Image(rgb_image))

#------------------------------------Chunk, For Multiprocessing-------------------------
def chunk_list(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

#------------------------------------Detect bad result-----------------------------------
def detect_bad_result(folder):
    fit = np.load(f"combined_data/{folder}/fitness.npy")
    bad = []
    q1, q3 = np.percentile(fit, [25, 75])
    iqr = q3 - q1
    # calculate the lower and upper bounds for outliers
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    # identify the outliers and their indices
    outliers = []
    outlier_indices = []
    medians = []
    for num in range(len(fit)): 
        # Load the two images and convert them to grayscale
        img1 = cv2.imread(f"combined_data/{folder}/color/{num:03}_mech.png",cv2.IMREAD_GRAYSCALE) # queryImage
        img2 = cv2.imread(f"combined_data/{folder}/color/{num:03}_real.png",cv2.IMREAD_GRAYSCALE) # trainImage
        # Initiate SIFT detector
        sift = cv2.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)
        good = []
        # ratio test as per Lowe's paper
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)
        src_pts = [ kp1[m.queryIdx] for m in good ]
        dst_pts = [ kp2[m.trainIdx] for m in good ]
        distance = [np.sqrt((kp1[m.queryIdx].pt[0] - kp2[m.trainIdx].pt[0])**2 + (kp1[m.queryIdx].pt[1] - kp2[m.trainIdx].pt[1])**2) for m in good]
        # np.save("distance.npy",distance)
        img1_with_keypoints = cv2.drawKeypoints(img1, src_pts, None)
        # cv.imshow("edge detection",img_with_keypoints)
        # cv.waitKey(0)
        img2_with_keypoints = cv2.drawKeypoints(img2, dst_pts, None)
        # cv.imshow("edge detection2",img_with_keypoints)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        stacked_img = np.hstack((img1_with_keypoints, img2_with_keypoints))
        # cv.namedWindow(f"edge detection {num:03}", cv.WINDOW_NORMAL)
        # cv.resizeWindow(f"edge detection {num:03}", 1920, 1080)
        # cv.imshow(f"edge detection {num:03}",stacked_img)
        # cv.waitKey(1000)
        # cv.destroyAllWindows()
        # plot the candlestick chart
        fig, ax = plt.subplots()
        bp = ax.boxplot(distance, vert=False, widths=0.5, whiskerprops=dict(linestyle='--'), showmeans=True)
        # Get the statistics for the boxplot

        # Access the statistics of the boxplot
        median = [item.get_xdata()[0] for item in bp['medians']]
        medians.append(median[0])
        print(f'{num:03}\tMedians: {median[0]}\n'
            f'\tFitness: {fit[num]}\n')
        
        alpha = 0.5
        beta = 1 - alpha
        # Blend the two images together
        blurred1 = cv2.GaussianBlur(img1, (5, 5), 0)
        edge1 = cv2.Canny(blurred1, 50, 200)
        blurred2 = cv2.GaussianBlur(img2, (5, 5), 0)
        edge2 = cv2.Canny(blurred2, 50, 200)
        result = cv2.addWeighted(edge1, alpha, edge2, beta, 0)
        cv2.imwrite(f"combined_data/{folder}/edge/{num:03}.png",result)
        # cv2.imshow(f"median: {medians[0]}",result)
        # cv2.waitKey(1000)
        # cv2.destroyAllWindows()

        plt.close()
        # plt.show(block=False)
        # plt.pause(1)
        # # plt.waitforbuttonpress()
        # plt.close()
        # if fit[num] < lower_bound:# or fit[num] > upper_bound:
        if median[0] > 4.0:
            outliers.append(fit[num])
            outlier_indices.append(f"{num:03}")

    # if medians[0] > 3:
    #     bad.append(num)
    print("The outliers in the data are:", outliers)
    print("The indices of the outliers are:", outlier_indices)
    np.save(f"combined_data/{folder}/outlier.npy", outlier_indices)
    np.save(f"combined_data/{folder}/medians.npy", medians)
    print(bad)

#------------------------------------Main----------------------------------------------
def main():
    start_time = time()
    for folder in scan_dir():
        fitness = []
        inlier_rmse = []
        transformation = []
        make_dir(folder)
        mech_poses = np.load(f"stored_data_mechmind/{folder}/tf_mechmind_to_baselink.npy")
        real_poses = np.load(f"stored_data_realsense/{folder}/tf_realsense_to_baselink.npy")
        #Project PointCloud to Depth/Color using Multiprocessing
        chunks = chunk_list(mech_poses, num_processes)
        e = 0
        for t_pro in chunks: # Multiprocessing
            processs_1 = []
            processs_2 = []
            for u,trans in enumerate(t_pro): 
                i = e + u
                print(f"Frame {i} | {folder}")
                pcd1 = process_mechmind_image(i, folder, mech_poses, crop=True)
                # pcd1.transform(trans)
                pcd2 = process_realsense_image(i, folder, real_poses, crop=True)
                # o3d.visualization.draw_geometries([pcd1, pcd2])
                source = pcd1.voxel_down_sample(voxel_size = 0.02)
                target = pcd2.voxel_down_sample(voxel_size = 0.02)
                tf_param, _, _ = cpd.registration_cpd(source, target, update_scale=False)
                # tf_param, _, _ = filterreg.registration_filterreg(source, target, objective_type = "pt2pt")
                pcd1.points = tf_param.transform(pcd1.points)
                # pcd1 = result.transform(np.linalg.inv(real_poses[i]))
                # pcd2.transform(np.linalg.inv(real_poses[i]))
                # o3d.visualization.draw_geometries([pcd1, pcd2])
                pcd1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
                pcd2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
                ref = refine_registration(pcd1, pcd2, 0.01, np.identity(4))
                print(f"Fitness: {ref[1].fitness} \n", )
                fitness.append(ref[1].fitness)
                pcd1 = process_mechmind_image(i, folder, mech_poses, crop=False)
                pcd1.points = tf_param.transform(pcd1.points)
                pcd1.transform(ref[0].transformation)
                pcd1.transform(np.linalg.inv(real_poses[i]))
                # pcd2 = process_realsense_image(i, folder, real_poses, crop=False)
                # pcd2.transform(np.linalg.inv(real_poses[i]))
                # o3d.visualization.draw_geometries([pcd1, pcd2])
                process_1 = Process(target=point_cloud_to_rgb_image, args=(pcd1, real_intrinsic_value, i, "mech"))
                processs_1.append(process_1)
                process_2 = Process(target=point_cloud_to_depth_image, args=(pcd1, real_intrinsic_value, i, "mech"))
                processs_2.append(process_2)
            icp_time = time() - start_time
            np.save(f"{path_image_folder}/fitness.npy", fitness)
            np.save(f"{path_image_folder}/mech_intrinsic_value.npy", mech_intrinsic_value.intrinsic_matrix)  
            np.save(f"{path_image_folder}/real_intrinsic_value.npy", real_intrinsic_value.intrinsic_matrix)

            for process in processs_1: 
                process.start()

            # Wait for all the processs to complete
            for process in processs_1: 
                process.join()
            
            for process in processs_2: 
                process.start()
            
            for process in processs_2: 
                process.join()

            e += num_processes

        project_time = time() - start_time - icp_time 
        detect_bad_result(folder)
        detect_time = time() - start_time - icp_time - project_time
    print(f"Registration:\t {icp_time}")
    print(f"Projection:\t {project_time}")
    print(f"Detection:\t {detect_time}")
    print(f"Total Time Taken:{time() - start_time}")

if __name__ == "__main__":
    num_processes = cpu_count()
    box = o3d.geometry.AxisAlignedBoundingBox([-0.13, 0.30, -0.05], [0.25, 0.65, 0.20])
    # box = o3d.geometry.AxisAlignedBoundingBox([-0.03, 0.27, 0.01], [0.15, 0.50, 0.20])
    #---------------------------------Mechmind------------------------------------------
    mech_cali_param = [1742.2480444837477, 1743.3689879936705, 632.2929090490464, 491.18813700971344, 1000.0]
    mech_intrinsic_value = o3d.camera.PinholeCameraIntrinsic(width=1280, height=1024, fx = mech_cali_param[0], fy = mech_cali_param[1], cx = mech_cali_param[2], cy = mech_cali_param[3])
    #---------------------------------RealSense-----------------------------------------
    real_cali_param = [637.5080495052193, 637.707917930532, 646.778180811219, 364.5848168512465, 1000.0]
    real_intrinsic_value = o3d.camera.PinholeCameraIntrinsic(width=1280, height=720, fx = real_cali_param[0], fy = real_cali_param[1], cx = real_cali_param[2], cy = real_cali_param[3])
    delete_folder("C_030")
    main()
    delete_folder("C_030")
    # detect_bad_result("150523_1212H")
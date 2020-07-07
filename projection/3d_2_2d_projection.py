import os
import h5py
import pcl
import numpy as np
from numpy.linalg import inv
import cv2
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R


# Extract file's information such as IDs and times
def extract_file_inf(file_name):
    file_name = file_name.split(".")[0]
    sensor_id, time = file_name.split("_")
    return sensor_id, time


# Decide whether the image time is near to lidar time
def find_nearest_time(im_time, lidar_time_list):
    min_time_distance = abs(im_time - lidar_time_list[0][0])
    min_lidar_time_index = 1

    for time_index in range(len(lidar_time_list)):
        lidar_time = lidar_time_list[time_index][0]
        time_distance = abs(im_time - lidar_time)
        if time_distance == 0:
            min_lidar_time_index = time_index
            break
        else:
            if time_distance > min_time_distance:
                continue
            elif time_distance < min_time_distance:
                min_time_distance = time_distance
                min_lidar_time_index = time_index

    return str(lidar_time_list[min_lidar_time_index][0])


# Find pose data according to time stamp data
def find_pose_data(db, stamp_name, _time):
    for t_index in range(len(db[stamp_name])):
        if str(db[stamp_name][t_index][0]) == _time:
            break

    return t_index


# Finding right camera intrinsic parameters according to camera id
def find_cam_int_param(cam_id, cam_param_list):
    for cam_pram_lable in cam_param_list:
        if cam_pram_lable[0] == cam_id:
            break

    return cam_pram_lable


# Make Transformation Matrix using rotation and translation
def rot_2_trans_mat(trans_parameter, rot_mat):
    trans_mat = np.identity(4)
    trans_mat[:3, :3] = rot_mat
    trans_mat[:3, 3] = trans_parameter

    return trans_mat


# Make calibration matrix
def calib_matrix_cal(f_x, f_y, c_x, c_y):

    cal_mat = np.identity(3)
    cal_mat[0, 0] = f_x
    cal_mat[1, 1] = f_y
    cal_mat[0, 2] = c_x
    cal_mat[1, 2] = c_y

    return cal_mat


# Calculate the 3d to 2d transformation and distortion
def distortion(pixel, k_1, k_2, k_3, p_1, p_2):

    X = pixel[:, 0]
    Y = pixel[:, 1]

    R2 = X*X + Y*Y
    R4 = R2*R2
    R6 = R2*R4


    dist_x = X*(1+ k_1*R2 + k_2*R4 + k_3*R6) + 2*p_1*X*Y + p_2*(R2 + 2*(X**2))
    dist_y = Y*(1+ k_1*R2 + k_2*R4 + k_3*R6) + p_1*(R2 + 2*(Y**2)) + 2*p_2*X*Y

    # Make homogeneous 2d parameters
    dist = np.concatenate((np.array([dist_x]), np.array([dist_y])), axis = 0)

    return dist.T




"""
Load images, database, point cloud files

First set the image, then find the lidar file that has nearest time stamp
To find nearest time stamp, get differences between image and lidar file's time stamp
Then extract minimum time difference

According to camera id, lidar id and time stamps, find the pose data from DB that stored in HDF5 file
"""

# 1. Directory setting
default_path = "/home/dnjswo0205/Desktop/dataset/b1/train/2019-04-16_15-35-46"
pc_path = os.path.join(default_path, "pointclouds_data")
im_path = os.path.join(default_path, "images")


# 2. Make lists of point cloud and image data
pc_list = os.listdir(pc_path)
im_list = os.listdir(im_path)


# 3. Set camera id and time as apart
tar_image = im_list[2500]
camera_ID, Im_time = extract_file_inf(tar_image)


# 4. Set the lidar ids
lidar_0_ID = "lidar0"
lidar_1_ID = "lidar1"


# 5. Load the database file
database = h5py.File(os.path.join(default_path, "groundtruth.hdf5"), "r")


# 6. Make lidar 0's time list
time_list = database["lidar0_stamp"][:]


# 7. Find the nearest lidar time stamp according to the image
min_lidar_time = find_nearest_time(int(Im_time), time_list)


# 8. Load the lidar 0 and lidar 1 point cloud files
pc_file_name = lidar_0_ID + "_" + min_lidar_time + ".pcd"
point_cloud = pcl.load(os.path.join(pc_path, pc_file_name))
pc = np.array(point_cloud)


# 9. Load the camera pose data from DB
# 1) Set the name of the camera pose data
c_stamp_name = camera_ID + "_stamp"
c_pose_name = camera_ID + "_pose"


# 2) Extract the camera pose parameter according to time stamp index
ct_index = find_pose_data(database, c_stamp_name, Im_time)
camera_pose_param = database[c_pose_name][ct_index]


# 10. Load the lidar pose data from DB
# 1) Set the name of the lidar pose data
l_0_stamp_name = lidar_0_ID + "_stamp"
l_0_pose_name = lidar_0_ID + "_pose"

# 2) Extract the lidar pose parameter according to time stamp index
l_0_t_index = find_pose_data(database, l_0_stamp_name, min_lidar_time)
lidar_0_pose_param = database[l_0_pose_name][l_0_t_index]


# 11. Load the camera intrinsic parameters from text file
# 1) Load the text file that contains camera intrinsic parameters
camera_para_name = "camera_parameters.txt"
text = open(os.path.join(default_path, camera_para_name), "r")


# 2) Read all the camera intrinsic parameters and make them as a list
camera_parameters = []

lines = text.readlines()
for line in lines[4:]:
    camera_parameters.append(line.split(" "))
text.close()


# 3) Extract camera information from camera_parameters
cam_int_params = find_cam_int_param(camera_ID, camera_parameters)
sx, sy, fx, fy, cx, cy, k1, k2, p1, p2, k3 = map(float, cam_int_params[1:])




"""
Calculate the projection from 3D point cloud to 2D image

First transfrom lidar frame points to camera frame points using extrinsic matrices(Transformation Matrix)
Then project camera frame 3D points onto 2D image using intrinsic matrices
In this process, calculate distortion to enhance the accuracy
"""


# 1. Transform lidar frame points to camera frame points using extrinsic matrices
# 1) Calculate rotation matrix using quaternion parameters
cqw, cqx, cqy, cqz = camera_pose_param[3:]
lqw, lqx, lqy, lqz = lidar_0_pose_param[3:]
c_rotation = R.from_quat([cqx, cqy, cqz, cqw]).as_dcm()
l_0_rotation = R.from_quat([lqx, lqy, lqz, lqw]).as_dcm()


# 2) Make transformation matrix using rotation and translation matrix
c_trans = rot_2_trans_mat(camera_pose_param[:3], c_rotation)
l_0_trans = rot_2_trans_mat(lidar_0_pose_param[:3], l_0_rotation)


# 3) Transform 3d points matrix into homogeneous shape
pc_homo = np.c_[pc, np.ones([len(pc), 1])]


# 4) Transform lidar frame points to world frame point using lidar transformation matrix
l_0_pc = np.matmul(l_0_trans, pc_homo.T).T
print(len(pc))

# 5) Transform world frame points to camera frame point using inversed camera transformation matrix
c_l_0_pc = np.matmul(inv(c_trans), l_0_pc.T).T


# 2. Project the 3D camera frame points onto the image
# 1) Set all the intrinsic matrices such as projection and calibration
proj_mat = np.c_[np.identity(3), np.zeros([3, 1])]
calib_mat = calib_matrix_cal(fx, fy, cx, cy)


# 2) Calculate the camera calibration using intrinsic matrices
prj_pc = np.matmul(proj_mat, c_l_0_pc.T).T
calib_pc = np.matmul(calib_mat, prj_pc.T).T


# 3) Extract the points contain positive integer of z(z+)
pos_z = np.where(calib_pc[:, 2] > 0)
front_pc = calib_pc[pos_z]


# 4) Calculate pixel coordinate system by dividing z and distortion
pix_point = front_pc/front_pc[:, 2][:, None]

dist_pix_point = distortion(pix_point, k1, k2, k3, p1, p2)


# 5) Extract pixel points inside of image frame
inside = np.logical_and(np.logical_and(dist_pix_point[:, 0] >= 0, dist_pix_point[:, 1] >= 0),
                        np.logical_and(dist_pix_point[:, 0] < sx, dist_pix_point[:, 1] < sy))

prj_pix_points = dist_pix_point[inside]




"""
Visualize the projected 2D points to check whether be projected well

First upload the target image using OpenCV library
Then copy the image data and draw points on it
"""
# 1. Load the image using OpenCV library
image = cv2.imread(os.path.join(im_path, tar_image), cv2.IMREAD_UNCHANGED)


# 2. Copy the target image
image_clone = image.copy()


# 3. Draw the points on copied image data
for point_idx in prj_pix_points:
    image_clone = cv2.line(image_clone, (int(round(point_idx[0])), int(round(point_idx[1]))),
                           (int(round(point_idx[0])), int(round(point_idx[1]))), (255, 0, 0), 5)



plt.figure(1)
plt.imshow(image_clone)
plt.show()
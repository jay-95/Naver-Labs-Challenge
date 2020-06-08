import os
import h5py
import pcl
import numpy as np
import cv2
from matplotlib import pyplot as plt


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


# Make Rotation Matrix using quaternion
def quat_2_rot_mat(parameter):
    trans_vec = parameter[:3]
    qw, qx, qy, qz = parameter[3:]

    xx = qx**2
    xy = qx*qy
    xz = qx*qz
    xw = qx*qw

    yy = qy**2
    yz = qy*qz
    yw = qy*qw

    zz = qz**2
    zw = qz*qw

    rot_matrix = np.array([[1 - 2 * (yy + zz), 2 * (xy - zw), 2 * (xz + yw)],
                           [2 * (xy + zw), 1 - 2 * (xx + zz), 2 * (yz - xw)],
                           [2 * (xz - yw), 2 * (yz + xw), 1 - 2 * (xx + yy)]])

    rot_trans = np.identity(4)
    rot_trans[:3, 3] = trans_vec
    rot_trans[:3, :3] = rot_matrix

    return rot_trans


# Finding right camera parameters according to camera id
def find_cam_param(cam_id, cam_param_list):
    for cam_pram_lable in cam_param_list:
        if cam_pram_lable[0] == cam_id:
            break
    return cam_pram_lable


# Make camera matrix
def camera_matrix_cal(f_x, f_y, c_x, c_y):

    cam_mat = np.identity(3)
    cam_mat[0, 0] = f_x
    cam_mat[1, 1] = f_y
    cam_mat[0, 2] = c_x
    cam_mat[1, 2] = c_y

    return cam_mat


# Calculate the 3d to 2d transformation and distortion
def _3d_2_2d_proj(cl_rot_mat):
    X = cl_rot_mat[0]
    Y = cl_rot_mat[1]
    Z = cl_rot_mat[2]


    X1 = X/Z
    Y1 = Y/Z
    R2 = X1**2 + Y1**2
    R4 = R2**2
    R6 = R2*R4


    X2 = X1*(1+ k1*R2 + k2*R4 + k3*R6) + 2*p1*X1*Y1 + p2*(R2 + 2*(X1**2))
    Y2 = Y1*(1+ k1*R2 + k2*R4 + k3*R6) + p1*(R2 + 2*(Y1**2)) + 2*p2*X1*Y1

    # Make homogeneous 2d parameters
    pc_dist_proj = np.concatenate((np.array([X2]), np.array([Y2])), axis = 0)
    pc_dist_proj_homo = np.concatenate((pc_dist_proj, ones), axis = 0)

    return pc_dist_proj_homo


# Eliminate the points that is outside of the image
def el_out_point(_3d_proj_points, c_x, c_y):
    _2d_proj_points = []

    for _3d_proj_point in _3d_proj_points:
        if (0 <= _3d_proj_point[0] < 2*c_x) and (0 <= _3d_proj_point[1] < 2*c_y):
            _2d_proj_points.append([int(round(_3d_proj_point[0])), int(round(_3d_proj_point[1]))])

    return _2d_proj_points


# Directory setting
default_path = "/home/dnjswo0205/Desktop/dataset/b1/train/2019-04-16_15-35-46/"
pc_path = default_path + "pointclouds_data/"
im_path = default_path + "images/"

# Make pcd and image file lists
pc_list = os.listdir(pc_path)
im_list = os.listdir(im_path)

# Load the database file
database = h5py.File(default_path + "groundtruth.hdf5", "r")
# Make database keys into list to open the database
hdf_keys = list(database)


### from this part, should be modified
# image test part (can be deleted)
Image_test = im_list[0]
camera_ID, Im_time = extract_file_inf(Image_test)
time_array = database["lidar0_stamp"][:]

# find the nearest lidar time stamp according to the image
min_lidar_time = find_nearest_time(int(Im_time), time_array)

# load the lidar0 and lidar1 point cloud files
lidar0_ID = "lidar0"
lidar1_ID = "lidar1"
pc0_file_name = lidar0_ID + "_" + min_lidar_time + ".pcd"
pc1_file_name = lidar1_ID + "_" + min_lidar_time + ".pcd"
point_cloud0 = pcl.load(pc_path + pc0_file_name)
point_cloud1 = pcl.load(pc_path + pc1_file_name)
pc0 = np.array(point_cloud0)
pc1 = np.array(point_cloud1)

# load the camera pose data from DB
# set the name of the camera pose data
c_stamp_name = camera_ID + "_stamp"
c_pose_name = camera_ID + "_pose"

# extract the camera pose parameter according to time stamp index
ct_index = find_pose_data(database, c_stamp_name, Im_time)

camera_pose_param = database[c_pose_name][ct_index]

# load the lidar pose data from DB
l0_stamp_name = lidar0_ID + "_stamp"
l0_pose_name = lidar0_ID + "_pose"

l1_stamp_name = lidar1_ID + "_stamp"
l1_pose_name = lidar1_ID + "_pose"

l0t_index = find_pose_data(database, l0_stamp_name, min_lidar_time)
l1t_index = find_pose_data(database, l1_stamp_name, min_lidar_time)
print(Im_time, min_lidar_time)
lidar0_pose_param = database[l0_pose_name][l0t_index]
lidar1_pose_param = database[l1_pose_name][l1t_index]
print(l0t_index)
print(l1t_index)

# calculate rotation translation matrix using quaternion
c_rot_trans_matrix = quat_2_rot_mat(camera_pose_param)
l_rot_trans_matrix0 = quat_2_rot_mat(lidar0_pose_param)
l_rot_trans_matrix1 = quat_2_rot_mat(lidar1_pose_param)

### end of modifying area

# Load the camera parameters
camera_para_name = "camera_parameters.txt"
text = open(default_path + camera_para_name, "r")

camera_parameters = []

lines = text.readlines()
for line in lines[4:]:
    camera_parameters.append(line.split(" "))
text.close()

# Extract camera information from camera_parameters
cam_params = find_cam_param(camera_ID, camera_parameters)
sx, sy, fx, fy, cx, cy, k1, k2, p1, p2, k3 = map(float, cam_params[1:])

# Calculate the camera matrix
camera_matrix = camera_matrix_cal(fx, fy, cx, cy)

# Transform 3d points into homogeneous shape for rotate calculation
pc_trans0 = np.transpose(pc0)
ones = np.ones((1, len(pc0)))
pc_homo0 = np.concatenate((pc_trans0, ones), axis = 0)

pc_trans1 = np.transpose(pc1)
pc_homo1 = np.concatenate((pc_trans1, ones), axis = 0)

# Calculate the rotation and projection
# 1. Calculate the lidar rotation
l_pc_rot0 = np.dot(l_rot_trans_matrix0, pc_homo0)
l_pc_rot1 = np.dot(l_rot_trans_matrix1, pc_homo1)

# 2. Calculate the camera rotation with the inversed rotation matrix
cl_pc_rot0 = np.dot(np.linalg.inv(c_rot_trans_matrix), l_pc_rot0)
cl_pc_rot1 = np.dot(np.linalg.inv(c_rot_trans_matrix), l_pc_rot1)

# Calculate the 3d to 2d transformation and distortion
pc_dist_homo0 = _3d_2_2d_proj(cl_pc_rot0)
pc_dist_homo1 = _3d_2_2d_proj(cl_pc_rot1)


# Calculate projection
pc_proj0 = np.dot(camera_matrix, pc_dist_homo0)
feature_points0 = np.transpose(pc_proj0[:2])

pc_proj1 = np.dot(camera_matrix, pc_dist_homo1)
feature_points1 = np.transpose(pc_proj1[:2])

# Eliminate the points that is outside of the image
proj_2d_points0 = el_out_point(feature_points0, cx, cy)
proj_2d_points1 = el_out_point(feature_points1, cx, cy)
print(proj_2d_points0)
# Load the image
Im_test = cv2.imread(im_path + Image_test, cv2.IMREAD_UNCHANGED)

# Copy the test image
Im_test_clone = Im_test.copy()

# Draw the points to Im_test_clone image according to the projected results
for point_idx0 in proj_2d_points0:
    Im_test_clone = cv2.line(Im_test_clone, (point_idx0[0], point_idx0[1]), (point_idx0[0], point_idx0[1]), (255, 0, 0), 5)

#for point_idx1 in proj_2d_points1:
#    Im_test_clone = cv2.line(Im_test_clone, (point_idx1[0], point_idx1[1]), (point_idx1[0], point_idx1[1]), (255, 0, 0), 5)

plt.figure(1)
plt.imshow(Im_test_clone)
plt.show()

print(im_list)
#print(hdf_keys)
#print(database[pose_name][t_index])
#print(Extract_file_inf(pc_list[0]))


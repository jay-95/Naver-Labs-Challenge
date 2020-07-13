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
def find_nearest_time(im_time, lidar_time_list, ts_range):

    lidar_times = []
    for time_index in range(len(lidar_time_list)):
        lidar_time = lidar_time_list[time_index][0]
        if ((im_time - ts_range) < lidar_time) and ((im_time + ts_range) > lidar_time):
            lidar_times.append(str(lidar_time))

    return lidar_times


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
    pixel[:, 0] = dist_x
    pixel[:, 1] = dist_y

    return pixel


def compute(qimage, dbimage, eps = 1e-7):
    extractor = cv2.xfeatures2d.SIFT_create()
    kps1, descs1 = extractor.detectAndCompute(qimage, None)
    kps2, descs2 = extractor.detectAndCompute(dbimage, None)

    descs1 /= (descs1.sum(axis = 1, keepdims = True) + eps)
    descs1 = np.sqrt(descs1)

    descs2 /= (descs2.sum(axis = 1, keepdims = True) + eps)
    descs2 = np.sqrt(descs2)

    return(kps1, descs1, kps2, descs2)



"""
NETVLAD
According to query image(test image), find the closest scenery among train images
"""


default_path = "/home/dnjswo0205/Desktop/dataset/b1"
test_path = os.path.join(default_path, "test/2019-08-21_09-49-05")
test_image_path = os.path.join(test_path, "images")
train_path = os.path.join(default_path, "train/2019-04-16_16-14-48")
train_image_path = os.path.join(train_path, "images")
pc_path = os.path.join(train_path, "pointclouds_data")


test_image_name = "40027089_1566000000001726.jpg"
train_image_name = "22970285_1555398949779869.jpg"


"""
Projection
Project 3D point clouds onto 2D camera
"""


camera_ID, Im_time = extract_file_inf(train_image_name)
lidar_IDs = ["lidar0", "lidar1"]


database = h5py.File(os.path.join(train_path, "groundtruth.hdf5"), "r")


c_stamp_name = camera_ID + "_stamp"
c_pose_name = camera_ID + "_pose"


ct_index = find_pose_data(database, c_stamp_name, Im_time)
camera_pose_param = database[c_pose_name][ct_index]


camera_para_name = "camera_parameters.txt"
text = open(os.path.join(train_path, camera_para_name), "r")


camera_parameters = []

lines = text.readlines()
for line in lines[4:]:
    if line[0] != " " and line[0] != "\n":
        camera_parameters.append(line[:-1].split(" "))
text.close()

cam_int_params = find_cam_int_param(camera_ID, camera_parameters)
sx, sy, fx, fy, cx, cy, k1, k2, p1, p2, k3 = map(float, cam_int_params[1:])

l_pc = []

for lidar_ID in lidar_IDs:

    # 6. Make lidar 0's time list
    time_list = database[lidar_ID + "_stamp"][:]

    # 7. Find the nearest lidar time stamp according to the image
    min_lidar_times = find_nearest_time(int(Im_time), time_list, 2000000)

    for idx in min_lidar_times:
        # 1) Load the lidar 0 and lidar 1 point cloud files
        pc_file_name = lidar_ID + "_" + idx + ".pcd"
        point_cloud = pcl.load(os.path.join(pc_path, pc_file_name))
        pc = np.array(point_cloud)
        invalid = np.logical_and(np.logical_and(pc[:, 0] == 0, pc[:, 1] == 0), pc[:, 2] == 0)
        pc = pc[~invalid]


        # 2) Load the lidar pose data from DB
        # 1) Set the name of the lidar pose data
        l_stamp_name = lidar_ID + "_stamp"
        l_pose_name = lidar_ID + "_pose"

        # 3) Extract the lidar pose parameter according to time stamp index
        l_t_index = find_pose_data(database, l_stamp_name, idx)
        lidar_pose_param = database[l_pose_name][l_t_index]

        # 4) Calculate rotation matrix using quaternion parameters
        lqw, lqx, lqy, lqz = lidar_pose_param[3:]
        l_rotation = R.from_quat([lqx, lqy, lqz, lqw]).as_matrix()
        l_trans = rot_2_trans_mat(lidar_pose_param[:3], l_rotation)

        # 5) Transform 3d points matrix into homogeneous shape
        pc_homo = np.c_[pc, np.ones([len(pc), 1])]

        # 6) Transform lidar frame points to world frame point using lidar transformation matrix
        l_pc.append(np.matmul(l_trans, pc_homo.T).T)


l_pc = np.concatenate(l_pc, axis=0)


# 1. Transform lidar frame points to camera frame points using extrinsic matrices
# 1) Calculate rotation matrix using quaternion parameters
cqw, cqx, cqy, cqz = camera_pose_param[3:]
c_rotation = R.from_quat([cqx, cqy, cqz, cqw]).as_matrix()


# 2) Make transformation matrix using rotation and translation matrix
c_trans = rot_2_trans_mat(camera_pose_param[:3], c_rotation)


# 5) Transform world frame points to camera frame point using inversed camera transformation matrix
c_l_pc = np.matmul(inv(c_trans), l_pc.T).T


# 3) Extract the points contain positive integer of z(z+)
pos_z = np.where(c_l_pc[:, 2] > 0)
c_l_pc = c_l_pc[pos_z]


# 2. Project the 3D camera frame points onto the image
# 1) Set all the intrinsic matrices such as projection and calibration
proj_mat = np.c_[np.identity(3), np.zeros([3, 1])]
calib_mat = calib_matrix_cal(fx, fy, cx, cy)


# 2) Calculate the camera calibration using intrinsic matrices
prj_pc = np.matmul(proj_mat, c_l_pc.T).T
prj_point = prj_pc/prj_pc[:, 2][:, None]


# 4) Calculate pixel coordinate system by dividing z and distortion
dist_pix_point = distortion(prj_point, k1, k2, k3, p1, p2)
calib_pc = np.matmul(calib_mat, dist_pix_point.T).T


# 5) Extract pixel points inside of image frame
inside = np.logical_and(np.logical_and(calib_pc[:, 0] >= 0, calib_pc[:, 1] >= 0),
                        np.logical_and(calib_pc[:, 0] < sx, calib_pc[:, 1] < sy))

prj_pix_points = calib_pc[inside]

l_pc = l_pc[pos_z][inside]


train_image = cv2.imread(os.path.join(train_image_path, train_image_name), cv2.IMREAD_UNCHANGED)
test_image = cv2.imread(os.path.join(test_image_path, test_image_name), cv2.IMREAD_UNCHANGED)


plt.figure(1)
plt.imshow(train_image)
plt.scatter(prj_pix_points[:, 0], prj_pix_points[:, 1], s=1, c = c_l_pc[inside][:,2], edgecolors='none')

plt.figure(2)
plt.imshow(train_image)

plt.figure(3)
plt.imshow(test_image)


"""
Root_SIFT
"""
kp1, des1, kp2, des2 = compute(test_image, train_image)
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
good = []


for m,n in matches:
    if m.distance < 0.6 * n.distance:
        good.append([m])


draw_image = cv2.drawMatchesKnn(test_image, kp1, train_image, kp2, good, None, flags = 2)

list_kp1 = []
list_kp2 = []

for g in good:
    test_idx = g[-1].queryIdx
    train_idx = g[-1].trainIdx

    (x1, y1) = kp1[test_idx].pt
    (x2, y2) = kp2[train_idx].pt

    list_kp1.append((x1, y1))
    list_kp2.append((x2, y2))

plt.figure(4)
plt.imshow(draw_image)


plt.show()

"""
Match 2D key points with 3D world frame points
"""

kp3d = []
kp2d = []
print(l_pc)

for kp_idx in range(len(list_kp2)):
    train_kpx = list_kp2[kp_idx][0]
    train_kpy = list_kp2[kp_idx][0]

    px = prj_pix_points[:, 0]
    py = prj_pix_points[:, 1]

    dis_x = px - train_kpx
    dis_y = py - train_kpy

    min_dis = np.sqrt(dis_x*dis_x + dis_y*dis_y)

    if min(min_dis) >= 1:
        continue

    min_idx = np.argmin(min_dis)
    kp3d.append(l_pc[min_idx][:3])
    kp2d.append(list_kp1[kp_idx])

kp3d = np.array(kp3d)
kp2d = np.array(kp2d)


"""
Solve PnP
"""
ttext = open(os.path.join(test_path, camera_para_name), "r")


tcamera_parameters = []

tlines = ttext.readlines()
for tline in tlines[4:]:
    if tline[0] != " " and tline[0] != "\n":
        tcamera_parameters.append(tline[:-1].split(" "))
ttext.close()

tcamera_ID, _ = extract_file_inf(test_image_name)

tcam_int_params = find_cam_int_param(tcamera_ID, tcamera_parameters)
tsx, tsy, tfx, tfy, tcx, tcy, tk1, tk2, tp1, tp2, tk3 = map(float, tcam_int_params[1:])


distCoeffs = np.array([tk1, tk2, tp1, tp2, tk3])
_, rvec, tvec, _ = cv2.solvePnPRansac(kp3d, kp2d, calib_mat, distCoeffs)
rmat = R.from_rotvec([rvec[0][0], rvec[1][0], rvec[2][0]]).as_matrix()

est_rmat, _ = cv2.Rodrigues(rvec, rmat)

est_quat = R.from_matrix(inv(est_rmat)).as_quat()
ext_tran = np.matmul(-inv(est_rmat), tvec)


print(est_quat)
print(cqx,cqy,cqz,cqw)
print(ext_tran)
print(camera_pose_param[:3])
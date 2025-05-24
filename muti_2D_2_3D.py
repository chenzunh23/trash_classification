import json
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
group_number = 45
image_number_each_group = 4
image_folder = 'label_images/images'
focal_length = 24  # mm
image_width = 4096
image_height = 3072
cameras_position = [(0, -75, 50), (45, -45, 50),
                    (0, 75, 50), (-45, 45, 50)]  # cm
cameras_rotate_matrix = [[[1, 0, 0],
                          [0, 1/np.sqrt(2), -1/np.sqrt(2)],
                          [0, 1/np.sqrt(2), 1/np.sqrt(2)]],
                         [[1/np.sqrt(2), 1/np.sqrt(2), 0],
                          [-1/2, 1/2, -1/np.sqrt(2)],
                          [-1/2, 1/2, 1/np.sqrt(2)]],
                         [[-1, 0, 0],
                          [0, -1/np.sqrt(2), -1/np.sqrt(2)],
                          [0, -1/np.sqrt(2), 1/np.sqrt(2)]],
                         [[-1/np.sqrt(2), -1/np.sqrt(2), 0],
                          [1/2, -1/2, -1/np.sqrt(2)],
                          [1/2, -1/2, 1/np.sqrt(2)]]]


def get_camera_matrix():
    # 焦距(mm)转换为像素
    fx = focal_length * image_width / 36  # 假设传感器宽度为36mm
    fy = focal_length * image_height / 24  # 假设传感器高度为24mm
    cx = image_width / 2
    cy = image_height / 2
    return np.array([[fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]])


def project_2d_to_3d(points_2d, camera_idx):
    # 获取相机参数
    camera_matrix = get_camera_matrix()
    R = np.array(cameras_rotate_matrix[camera_idx])
    t = np.array(cameras_position[camera_idx])

    # 归一化坐标
    points_2d_homo = np.hstack((points_2d, np.ones((len(points_2d), 1))))
    normalized_points = np.linalg.inv(camera_matrix) @ points_2d_homo.T

    # 转换到世界坐标系
    points_3d = []
    for point in normalized_points.T:
        # 使用射线-平面相交计算3D点
        direction = R.T @ point
        origin = -R.T @ t

        # 假设物体在平面z=0上
        t = -origin[2] / direction[2]
        point_3d = origin + t * direction
        points_3d.append(point_3d)

    return np.array(points_3d)


with open('label_images/result.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
categories = data['categories']  # 获取类别信息
# example  categories[0]={
#   "id": 0,
#   "name": "Aerosol"
# }

segmentation = {}
for group in range(group_number):
    for image_index in range(image_number_each_group):
        image_name = f"{group}_{image_index}.jpg"
        image_path = os.path.join(image_folder, image_name)
        if os.path.exists(image_path):
            segmentation[image_name] = {}
            for image in data['images']:
                if image_name in image['file_name']:
                    image_id = image['id']
                    break
            for ann in data['annotations']:
                if ann['image_id'] == image_id:
                    category_id = ann['category_id']
                    # 如果类别ID不存在，初始化为空列表
                    if category_id not in segmentation[image_name]:
                        segmentation[image_name][category_id] = []

                    segmentation_points = []
                    seg = ann['segmentation']
                    # If segmentation is a list of lists, flatten it
                    if isinstance(seg, list) and len(seg) > 0 and isinstance(seg[0], list):
                        seg = seg[0]
                    for i in range(0, len(seg), 2):
                        x = seg[i]
                        y = seg[i + 1]
                        segmentation_points.append((x, y))
                    segmentation[image_name][category_id].append(
                        segmentation_points)


def reconstruct_3d_objects(group_num=10):
    objects_3d = {}

    # 处理该组的4张图片
    for image_idx in range(image_number_each_group):
        image_name = f"{group_num}_{image_idx}.jpg"
        if image_name in segmentation:
            # 对每个类别的物体进行处理
            for category_id, segments in segmentation[image_name].items():
                if category_id not in objects_3d:
                    objects_3d[category_id] = []

                # 处理每个分割区域
                for segment_points in segments:
                    points_2d = np.array(segment_points)
                    points_3d = project_2d_to_3d(points_2d, image_idx)
                    objects_3d[category_id].append(points_3d)

    return objects_3d


# 重建3D物体
objects_3d = reconstruct_3d_objects(10)

# 可视化结果
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

for category_id, object_points in objects_3d.items():
    category_name = next(cat['name']
                         for cat in categories if cat['id'] == category_id)
    for points in object_points:
        ax.scatter(points[:, 0], points[:, 1],
                   points[:, 2], label=category_name)

ax.set_xlabel('X (cm)')
ax.set_ylabel('Y (cm)')
ax.set_zlabel('Z (cm)')
plt.title('3D Reconstruction of Objects in Group 10')
plt.legend()
plt.show()

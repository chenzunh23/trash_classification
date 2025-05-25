from mpl_toolkits.mplot3d import Axes3D
import json
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
group_number = 45
image_number_each_group = 4
image_folder = 'GroupedPhotos'
focal_length = 5.71  # mm
image_width = 4096
image_height = 3072
cameras_position = [(0, -75, 50), (45, -45, 50),
                    (0, 75, 50), (-45, 45, 50)]  # cm
cameras_rotate_matrix = [
    [[-1/np.sqrt(2), -1/np.sqrt(2), 0],
     [1/2, -1/2, 1/np.sqrt(2)],
     [1/2, -1/2, -1/np.sqrt(2)]],
    [[-1, 0, 0],
     [0, -1/np.sqrt(2), 1/np.sqrt(2)],
     [0, -1/np.sqrt(2), -1/np.sqrt(2)]],
    [[1/np.sqrt(2), 1/np.sqrt(2), 0],
     [-1/2, 1/2, 1/np.sqrt(2)],
     [-1/2, 1/2, -1/np.sqrt(2)]],
    [[1, 0, 0],
     [0, 1/np.sqrt(2), 1/np.sqrt(2)],
     [0, 1/np.sqrt(2), -1/np.sqrt(2)]]]


def get_camera_matrix():
    # 焦距(mm)转换为像素
    fx = focal_length * image_width / 8.21
    fy = focal_length * image_height / 6.16
    cx = image_width / 2
    cy = image_height / 2
    return np.array([[fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]])


def get_frustum_planes(points_2d, R, t, camera_matrix):
    """计算视锥体的平面方程"""
    points_2d = np.array(points_2d)
    planes = []

    # 对轮廓上的每相邻两点构造平面
    for i in range(len(points_2d)):
        p1 = points_2d[i]
        p2 = points_2d[(i + 1) % len(points_2d)]

        # 计算相机坐标系下的射线方向
        ray1 = np.linalg.inv(camera_matrix) @ np.array([p1[0], p1[1], 1])
        ray2 = np.linalg.inv(camera_matrix) @ np.array([p2[0], p2[1], 1])

        # 转换到世界坐标系
        ray1_world = R.T @ ray1
        ray2_world = R.T @ ray2

        # 计算平面法向量（两条射线和相机位置确定一个平面）
        normal = np.cross(ray1_world, ray2_world)
        normal = normal / np.linalg.norm(normal)

        # 平面方程：ax + by + cz + d = 0
        d = -np.dot(normal, t)
        planes.append(np.append(normal, d))

    return planes


def find_plane_intersections(planes):
    """计算平面的交点"""
    vertices = []
    n = len(planes)
    # 任选三个平面求交点
    # 定义不同的阈值
    thresholds = [40, 50, 55, 60, 70]
    vertices_dict = {th: [] for th in thresholds}

    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                try:
                    # 解线性方程组求交点
                    A = np.vstack(
                        [planes[i][:3], planes[j][:3], planes[k][:3]])
                    b = np.array([-planes[i][3], -planes[j][3], -planes[k][3]])
                    point = np.linalg.solve(A, b)

                    # 对每个阈值分别判断
                    for th in thresholds:
                        valid = True
                        for p in planes:
                            if np.dot(p[:3], point) + p[3] > th:
                                valid = False
                                break
                        if valid:
                            vertices_dict[th].append(point)
                except np.linalg.LinAlgError:
                    continue

    # 输出最严格的有至少5个点的一组
    for th in thresholds:
        print(f"阈值{th}下的交点数: {len(vertices_dict[th])}")
        if len(vertices_dict[th]) >= 5:
            vertices = vertices_dict[th]
            print(f"采用阈值{th}，共获得{len(vertices)}个交点")
            break
    else:
        # 如果都不满足，返回最大阈值的结果
        vertices = vertices_dict[thresholds[-1]]
        print(f"所有阈值下交点数均不足5,采用最大阈值{thresholds[-1]},共获得{len(vertices)}个交点")

    return np.array(vertices)


def project_2D_to_3D(segmentation, group_num, category_id):
    """
    将2D分割重建为3D物体
    """
    # 收集有效视图的2D点
    valid_views = []
    points_2d_all = []
    for image_index in range(image_number_each_group):
        image_name = f"{group_num}_{image_index}.jpg"
        if image_name in segmentation and category_id in segmentation[image_name]:
            valid_views.append(image_index)
            points = segmentation[image_name][category_id]
            if len(points) > 50:  # 修改判断条件
                idx = np.random.choice(len(points), 50, replace=False)
                points = points[idx]
            points_2d_all.append(points)

    # 如果有效视图少于2个，无法重建
    if len(valid_views) < 2:
        return []

    # 收集所有视锥体的平面
    all_planes = []
    for view_idx, points_2d in zip(valid_views, points_2d_all):
        R = np.array(cameras_rotate_matrix[view_idx])
        t = np.array(cameras_position[view_idx])
        camera_matrix = get_camera_matrix()
        print(camera_matrix)

        # 获取该视角的视锥体平面
        planes = get_frustum_planes(points_2d, R, t, camera_matrix)
        all_planes.extend(planes)

    # 计算所有平面的交点
    vertices = find_plane_intersections(all_planes)

    if len(vertices) < 4:
        print(f"在组{group_num}的类别{category_id}中,无法找到足够的3D点进行重建。")
        return []

    # 使用凸包算法构建最终的3D形状
    try:
        from scipy.spatial import ConvexHull

        hull = ConvexHull(vertices)
        final_vertices = vertices[hull.vertices]
        return [final_vertices]
    except Exception as e:
        print(f"无法计算凸包: {str(e)}")
        return [vertices]


with open('data/annotations_taco.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
categories = data['categories']  # 获取类别信息
# example  categories[0]={
#   "id": 0,
#   "name": "Aerosol"
# }

segmentation = {}
for group in range(group_number):
    for image_index in range(image_number_each_group):
        image_group = f"Group_{group}"
        image_name = f"{group}_{image_index}.jpg"
        image_path = os.path.join(image_folder, image_group, image_name)
        if os.path.exists(image_path):
            segmentation[image_name] = {}
            for image in data['images']:
                if image_name in image['file_name']:
                    image_id = image['id']
                    break
            for ann in data['annotations']:
                if ann['image_id'] == image_id and ann['segmentation'] != []:
                    category_id = ann['category_id']
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
                    segmentation[image_name][category_id] = np.array(
                        segmentation_points)


def reconstruct_3D_objects(group_num=11):
    objects_3D = {}

    # 处理该组的4张图片
    for image_idx in range(image_number_each_group):
        image_name = f"{group_num}_{image_idx}.jpg"
        if image_name not in segmentation:
            return None
    for category_id in range(len(categories)):
        objects_3D[category_id] = []
        objects_3D[category_id] = project_2D_to_3D(
            segmentation, group_num, category_id)
    view_3D_objects(objects_3D, group_num)
    return objects_3D


def view_3D_objects(objects_3D, group_num):

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    for category_id, object_points in objects_3D.items():
        category_name = next(cat['name']
                             for cat in categories if cat['id'] == category_id)
        for points in object_points:
            if len(points) > 0:
                ax.plot_trisurf(points[:, 0], points[:, 1],
                                points[:, 2], alpha=0.5, label=category_name)
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=10)

    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_zlabel('Z (cm)')
    plt.title(f'3D Reconstruction of Objects in Group {group_num}')
    plt.legend()

    # 设置更偏俯视的视角
    ax.view_init(elev=45, azim=-90)  # elev越大越俯视，azim可调整方向

    plt.show()


reconstruct_3D_objects(11)

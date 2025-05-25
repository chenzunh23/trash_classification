from mpl_toolkits.mplot3d import Axes3D
import json
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from data.YOLO_data.format_yolo_data import yolo_to_taco
from data.YOLO_data.predict_seg import predict_seg_model
from vis_images import vis_image

import argparse

from ultralytics import YOLO

group_number = 45
image_number_each_group = 4
image_folder = 'GroupedPhotos'
focal_length = 5.71  # mm
image_width = 4096
image_height = 3072
cameras_position =  [(-45, 45, 50), (0, 75, 50), (45, -45, 50), (0, -75, 50)]# cm
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

def load_segmentation_data(config_json='data/annotations_taco.json'):
    with open(config_json, 'r', encoding='utf-8') as f:
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
                image_id = None
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
    return segmentation, categories


def reconstruct_3D_objects(group_num=11, show_3D=True, save_path=None, config_json='data/annotations_taco.json'):
    objects_3D = {}
    segmentation, categories = load_segmentation_data(config_json=config_json)
    # 处理该组的4张图片
    for image_idx in range(image_number_each_group):
        image_name = f"{group_num}_{image_idx}.jpg"
        if image_name not in segmentation:
            return None
    for category_id in range(len(categories)):
        objects_3D[category_id] = []
        objects_3D[category_id] = project_2D_to_3D(
            segmentation, group_num, category_id)
    view_3D_objects(objects_3D, group_num, categories, show_3D, save_path)
    return objects_3D


# def view_3D_objects(objects_3D, group_num, categories, show3D, save_path):

#     fig = plt.figure(figsize=(10, 10))
#     ax = fig.add_subplot(111, projection='3d')

#     for category_id, object_points in objects_3D.items():
#         category_name = next(cat['name']
#                              for cat in categories if cat['id'] == category_id)
#         for points in object_points:
#             if len(points) > 0:
#                 ax.plot_trisurf(points[:, 0], points[:, 1],
#                                 points[:, 2], alpha=0.5, label=category_name)
#                 ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=10)

#     ax.set_xlabel('X (cm)')
#     ax.set_ylabel('Y (cm)')
#     ax.set_zlabel('Z (cm)')
#     plt.title(f'3D Reconstruction of Objects in Group {group_num}')
#     plt.legend()

#     # 设置更偏俯视的视角
#     azims = np.linspace(0, 360, 8, endpoint=False)
#     for azim in azims:
#         ax.view_init(elev=45, azim=azim)  # elev越大越俯视，azim可调整方向
#         if save_path:
#             plt.savefig(os.path.join(save_path, f'3D_reconstruction_group_{group_num}_{azim}.jpg'), dpi=300)
#         if show3D:
#             plt.show()

def view_3D_objects(objects_3D, group_num, categories, show3D, save_path):
    # 生成数据集4个视角参数
    azims = [315, 270, 135, 90] 
    
    # ==================== 创建组合大图 ====================
    fig_combined, axs = plt.subplots(2, 2, figsize=(20, 20),
                            subplot_kw={'projection': '3d'})
    axs = axs.ravel()  # 将子图数组展平为一维
    
    # 遍历每个子图绘制
    for idx, (ax, azim) in enumerate(zip(axs, azims)):
        # 绘制3D对象
        for category_id, object_points in objects_3D.items():
            category_name = next(cat['name'] for cat in categories if cat['id'] == category_id)
            for points in object_points:
                if len(points) > 0:
                    # 仅在第一个子图显示图例
                    label = category_name if idx == 0 else ""
                    ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2],
                                   alpha=0.5, label=label)
                    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=10)
        
        # 设置视角和基础信息
        ax.view_init(elev=45, azim=azim)
        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        ax.set_zlabel('Z (cm)')
        ax.set_title(f'View Angle: {azim:.0f}°', fontsize=8)
    
    # 添加全局图例和标题
    handles, labels = axs[0].get_legend_handles_labels()
    fig_combined.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.95, 0.9))
    fig_combined.suptitle(f'3D Reconstruction of Objects in Group {group_num}', y=0.95)
    plt.tight_layout()
    
    # 保存组合图
    if save_path:
        combined_path = os.path.join(save_path, f'group_{group_num}_batch.jpg')
        fig_combined.savefig(combined_path, dpi=200, bbox_inches='tight')
    if show3D:
        plt.show()
    plt.close(fig_combined)
    
    # ==================== 保存单张视角图 ====================
    for azim in azims:
        fig_single = plt.figure(figsize=(15, 15))
        ax_single = fig_single.add_subplot(111, projection='3d')
        
        # 绘制完整内容
        for category_id, object_points in objects_3D.items():
            category_name = next(cat['name'] for cat in categories if cat['id'] == category_id)
            for points in object_points:
                if len(points) > 0:
                    ax_single.plot_trisurf(points[:, 0], points[:, 1], points[:, 2],
                                         alpha=0.5, label=category_name)
                    ax_single.scatter(points[:, 0], points[:, 1], points[:, 2], s=10)
        
        # 设置视角和标题
        ax_single.view_init(elev=45, azim=azim)
        ax_single.set_xlabel('X (cm)')
        ax_single.set_ylabel('Y (cm)')
        ax_single.set_zlabel('Z (cm)')
        ax_single.legend()
        plt.title(f'3D Reconstruction - Group {group_num} ({azim:.0f}°)')
        
        # 保存单图
        if save_path:
            single_path = os.path.join(save_path, f'group_{group_num}_{azim:.0f}.jpg')
            plt.savefig(single_path, dpi=300, bbox_inches='tight')
        if show3D:
            plt.show()
        plt.close(fig_single)


def demo(group, model='yolo11n-seg', show=True, show_3D=True, save_path=None):
    """
    演示函数，展示最优模型的重建效果

    :param group: 组号
    :param show: 是否显示图像
    :param save_path: 保存路径

    :return: None
    """
    # 预测分割
    model = YOLO(model)  # load an official detection model
    model.info()  # print model information
    predict_seg_model(model=model, source=f"{image_folder}/Group_{group}",
                      save=True, save_txt=True, show=show, conf=0.25, show_labels=True, show_conf=False, imgsz=640)

    # 格式化数据
    predict_cnt = 0
    for i in os.listdir("./runs/segment/"):
        if "predict" in i:
            predict_cnt += 1
    print(f"Predict count: {predict_cnt}")
    predict_dir = f"./runs/segment/predict{predict_cnt}/labels"
    yolo_to_taco("./data/annotations_taco.json", predict_dir, f"output{predict_cnt}.json")

    # 处理JSON数据
    with open(f"output{predict_cnt}.json", 'r', encoding='utf-8') as f:
        data = json.load(f)

    categories = data['categories']  # 获取类别信息
        
    # 可视化图像
    print(f"Visualizing the first image for group {group}...")
    vis_image(f"{group}_0.jpg", save_path, image_folder,
              f"output{predict_cnt}.json", show)

    # 重建3D物体
    print(f"Reconstructing 3D objects for group {group}...")
    objects_3D = reconstruct_3D_objects(group, show_3D=show_3D, save_path=save_path, config_json=f"output{predict_cnt}.json")
    if objects_3D is None:
        print(f"Group {group} has no valid images for reconstruction.")
        return
    print(f"3D reconstruction completed for group {group}.")
    
    if show_3D == False and save_path is None:
        print("[WARNING] Please set show_3D to True or provide a save_path to view or save the 3D reconstruction.")
        return
    view_3D_objects(objects_3D, group, categories, show_3D, save_path)

if __name__ == "__main__":
    
    # reconstruct_3D_objects(11)
    parser = argparse.ArgumentParser(description='3D Object Reconstruction')
    parser.add_argument('--group', type=int, default=11,
                        help='Group number to reconstruct')
    parser.add_argument('--model', type=str, default='yolo11n-seg.pt',
                        help='Path to the YOLOv8 segmentation model')
    parser.add_argument('--show', action='store_true',
                        help='Show the images')
    parser.add_argument('--no_show_3D', action='store_false',   
                        help='Don\'t show the 3D reconstruction')
    parser.add_argument('--save_path', type=str, default='',
                        help='Path to save the visualized images')
    parser.add_argument("--predict", action="store_true",
                        help="Predict with YOLOv8 segmentation model")
    args = parser.parse_args()
    if args.predict:
        demo(args.group, model=args.model, show=args.show,
             show_3D=args.no_show_3D, save_path=args.save_path)
    else:
        reconstruct_3D_objects(args.group, show_3D=args.no_show_3D, save_path=args.save_path)

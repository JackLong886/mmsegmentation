import os
import random
import secrets
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                             r"envs\infer\Lib\site-packages\torch\lib"))
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from fastimg_func import IMAGE3, img_resample, minmax_stretch2range
from lightglue import LightGlue, SuperPoint, rbd
import torch
from torchvision import transforms

from sklearn.cluster import KMeans, DBSCAN
from test_func.rpc.coord_trans import *


def find_dense_areas_and_centers(points, eps=50, min_samples=10):
    # 创建 DBSCAN 聚类模型
    db = DBSCAN(eps=eps, min_samples=min_samples)
    db.fit(points)
    labels = db.labels_
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    cluster_centers = []
    for i in range(n_clusters):
        # 选择属于同一聚类的点
        cluster_points = points[labels == i]
        # 计算均值作为聚类中心
        center = cluster_points.mean(axis=0)
        cluster_centers.append(center)

    return np.stack(cluster_centers), labels


def find_clusters(points, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(points)
    return kmeans.cluster_centers_


def vis_pts(pts):
    """

    Args:
        pts: (N, 2)

    Returns:

    """
    color = [
        'b', 'g', 'r', 'c', 'm', 'y', 'k',  # 基本颜色
        'green', 'red', 'cyan', 'magenta', 'yellow', 'black',  # 全名颜色
        'orange', 'purple', 'brown', 'pink', 'olive', 'beige', 'maroon'  # 其他常见颜色
    ]

    plt.figure(figsize=(8, 6))  # 设置图的大小
    if not isinstance(pts, list):
        plt.scatter(pts[:, 0], pts[:, 1], c='blue', marker='o')  # 绘制散点图
    else:
        plt.scatter(pts[0][:, 0], pts[0][:, 1], c='blue', marker='o')  # 绘制散点图
        for pt in pts[1:]:
            c = color.pop(random.randint(0, len(color) - 1))
            plt.scatter(pt[:, 0], pt[:, 1], c=c, marker='o')  # 绘制散点图
    plt.title('2D Points')  # 设置图的标题
    plt.xlabel('X Axis')  # 设置 x 轴标签
    plt.ylabel('Y Axis')  # 设置 y 轴标签
    plt.grid(True)  # 显示网格
    plt.show()  # 显示图形


def vis_pic(img_data):
    if not isinstance(img_data, list):
        fig, axs = plt.subplots(1, 1, figsize=(5, 5))  # 创建一行两列的子图
        axs[0].imshow(src_gray)
        axs[0].set_title('Image')
        axs[0].axis('off')
    else:
        n = len(img_data)
        fig, axs = plt.subplots(1, n, figsize=(n * 5, 5))  # 创建一行两列的子图
        for i, imdata in enumerate(img_data):
            axs[i].imshow(imdata)
            axs[i].set_title(f'Image {i}')
            axs[i].axis('off')

    plt.tight_layout()
    plt.show()


class LgMatcher:
    def __init__(self):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.lg_extractor = SuperPoint(max_num_keypoints=2048).eval().to(self.device)
        self.lg_matcher = LightGlue(features='superpoint').eval().to(self.device)

    def __call__(self, imgray1, imgray2):
        to_tensor = transforms.ToTensor()
        assert len(imgray1.shape) == 2 and len(imgray2.shape) == 2
        img1_gray, img2_gray = to_tensor(imgray1).unsqueeze(0), to_tensor(imgray2).unsqueeze(0)
        feats1 = self.lg_extractor.extract(img1_gray.to(self.device))
        feats2 = self.lg_extractor.extract(img2_gray.to(self.device))

        matches01 = self.lg_matcher({'image0': feats1, 'image1': feats2})
        feats1, feats2, matches01 = [rbd(x) for x in [feats1, feats2, matches01]]  # remove batch dimension
        kpts0, kpts1, matches = feats1['keypoints'], feats2['keypoints'], matches01['matches']
        m_kpts1, m_kpts2 = kpts0[matches[..., 0]].cpu().numpy(), kpts1[matches[..., 1]].cpu().numpy()
        return m_kpts1, m_kpts2


if __name__ == '__main__':
    ref_path = r"C:\Users\ROG\Desktop\TMP\test\crop_base-10.tif"
    src_path = r"C:\Users\ROG\Desktop\TMP\test\GF2_PMS1_E108.8_N23.0_20221018_L1A0006832687-PAN1_output_regis.tif"
    work_dir = r'C:\Users\ROG\Desktop\TMP\test\wk'
    lg = LgMatcher()
    res1 = 30
    src_resam_path1 = os.path.join(work_dir, f'src_resample{res1}.tif')
    ref_resam_path1 = os.path.join(work_dir, f'ref_resample{res1}.tif')
    if not os.path.exists(src_resam_path1):
        src_resam_path1 = img_resample(src_path, src_resam_path1, res=res1)
        ref_resam_path1 = img_resample(ref_path, ref_resam_path1, res=res1)

    src = IMAGE3(src_resam_path1, if_print=True)
    ref = IMAGE3(ref_resam_path1, if_print=True)
    src_gray, ref_gray = src.get_extent(), ref.get_extent()
    if len(src_gray.shape) == 3:
        src_gray = np.mean(src_gray, axis=0)
    if len(ref_gray.shape) == 3:
        ref_gray = np.mean(ref_gray, axis=0)
    src_gray = minmax_stretch2range(src_gray, (0, 255)).astype(np.uint8)
    ref_gray = minmax_stretch2range(ref_gray, (0, 255)).astype(np.uint8)
    src_mkpts, ref_mkpts = lg(src_gray, ref_gray)

    cluster_centers2, labels = find_dense_areas_and_centers(src_mkpts, eps=50, min_samples=10)
    # vis_pts([src_mkpts, cluster_centers2])
    geo_pts = []
    for pt in cluster_centers2:
        geo_pts.append(cr2geo(src.im_proj, src.im_geotrans, pt[0], pt[1]))
    del src, ref

    src = IMAGE3(src_path, if_print=True)
    ref = IMAGE3(ref_path, if_print=True)
    win_size = 1024
    win_size_half = win_size // 2
    src_extents = []
    for geo_pt in geo_pts:
        c, r = geo2cr(src.im_proj, src.im_geotrans, geo_pt[0], geo_pt[1])
        src_c, src_r = max(0, c - win_size_half), max(0, r - win_size_half)
        if src_c + win_size > src.im_width:
            src_c = src.im_width - win_size
        if src_r + win_size > src.im_height:
            src_r = src.im_height - win_size
        src_extents.append([src_c, src_r, win_size, win_size])



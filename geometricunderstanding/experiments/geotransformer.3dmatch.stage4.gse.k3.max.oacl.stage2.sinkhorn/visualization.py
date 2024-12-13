import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def load_data(file_path):
    """
    从指定的文件路径加载数据。
    """
    data = np.load(file_path)
    ref_corr_points = data['ref_corr_points']
    src_corr_points = data['src_corr_points']
    corr_scores = data['corr_scores']
    transform = data['transform']
    estimated_transform = data['estimated_transform']
    inlier_mask = data['inlier_mask']
    return ref_corr_points, src_corr_points, corr_scores, transform, estimated_transform, inlier_mask

def visualize_matches(ref_corr_points, src_corr_points, transform, inlier_mask):
    """
    可视化匹配点对，并标注 inliers 和 outliers。
    """
    src_transformed = (transform[:3, :3] @ src_corr_points.T + transform[:3, 3:4]).T

    ref_pcd = o3d.geometry.PointCloud()
    src_pcd = o3d.geometry.PointCloud()
    ref_pcd.points = o3d.utility.Vector3dVector(ref_corr_points)
    src_pcd.points = o3d.utility.Vector3dVector(src_transformed)

    ref_colors = np.zeros((len(ref_corr_points), 3))
    src_colors = np.zeros((len(src_corr_points), 3))

    ref_colors[inlier_mask] = [0, 1, 0]  # inliers: green
    src_colors[inlier_mask] = [0, 1, 0]
    ref_colors[~inlier_mask] = [1, 0, 0]  # outliers: red
    src_colors[~inlier_mask] = [1, 0, 0]

    ref_pcd.colors = o3d.utility.Vector3dVector(ref_colors)
    src_pcd.colors = o3d.utility.Vector3dVector(src_colors)

    o3d.visualization.draw_geometries([ref_pcd, src_pcd], window_name='Point Cloud Matches')

def plot_error_distribution(errors, title='Error Distribution'):
    """
    绘制误差分布的直方图。
    """
    plt.figure()
    plt.hist(errors, bins=50, color='blue', alpha=0.7, edgecolor='black')
    plt.title(title)
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.grid()
    plt.show()

def scatter_score_vs_error(corr_scores, errors):
    """
    绘制匹配分数与误差之间的关系。
    """
    plt.figure()
    plt.scatter(corr_scores, errors, alpha=0.6, c='blue', edgecolor='w', s=50)
    plt.title('Correspondence Score vs. Registration Error')
    plt.xlabel('Correspondence Score')
    plt.ylabel('Registration Error')
    plt.grid()
    plt.show()

def process_all_results(result_dir):
    """
    遍历指定目录下的所有 .npz 文件，并对每个文件进行可视化分析。
    """
    for file_name in os.listdir(result_dir):
        if file_name.endswith('.npz'):
            file_path = os.path.join(result_dir, file_name)
            print(f'Processing {file_name}...')
            ref_corr_points, src_corr_points, corr_scores, transform, estimated_transform, inlier_mask = load_data(file_path)

            # 计算配准误差
            src_transformed = (transform[:3, :3] @ src_corr_points.T + transform[:3, 3:4]).T
            errors = np.linalg.norm(ref_corr_points - src_transformed, axis=1)

            # 可视化匹配点
            visualize_matches(ref_corr_points, src_corr_points, transform, inlier_mask)

            # 绘制误差分布
            plot_error_distribution(errors, title=f'Error Distribution - {file_name}')

            # 绘制匹配分数与误差的关系
            scatter_score_vs_error(corr_scores, errors)

if __name__ == '__main__':
    # 设置结果目录路径
    result_dir = './results'
    process_all_results(result_dir)

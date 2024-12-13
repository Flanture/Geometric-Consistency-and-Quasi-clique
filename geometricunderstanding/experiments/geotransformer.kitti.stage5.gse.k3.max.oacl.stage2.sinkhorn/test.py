import argparse
import os.path as osp
import time
import os
import numpy as np
import open3d as o3d

from geotransformer.engine import SingleTester
from geotransformer.utils.common import ensure_dir, get_log_string
from geotransformer.utils.torch import release_cuda

from config import make_cfg
from dataset import test_data_loader
from loss import Evaluator
from model import create_model


class Tester(SingleTester):
    def __init__(self, cfg):
        super().__init__(cfg)

        # dataloader
        start_time = time.time()
        data_loader, neighbor_limits = test_data_loader(cfg)
        loading_time = time.time() - start_time
        message = f'Data loader created: {loading_time:.3f}s collapsed.'
        self.logger.info(message)
        message = f'Calibrate neighbors: {neighbor_limits}.'
        self.logger.info(message)
        self.register_loader(data_loader)

        # model
        model = create_model(cfg).cuda()
        self.register_model(model)

        # evaluator
        self.evaluator = Evaluator(cfg).cuda()

        # preparation
        self.output_dir = osp.join(cfg.feature_dir)
        ensure_dir(self.output_dir)

    def test_step(self, iteration, data_dict):
        output_dict = self.model(data_dict)
        return output_dict

    def eval_step(self, iteration, data_dict, output_dict):
        result_dict = self.evaluator(output_dict, data_dict)
        return result_dict

    def summary_string(self, iteration, data_dict, output_dict, result_dict):
        seq_id = data_dict['seq_id']
        ref_frame = data_dict['ref_frame']
        src_frame = data_dict['src_frame']
        message = f'seq_id: {seq_id}, id0: {ref_frame}, id1: {src_frame}'
        message += ', ' + get_log_string(result_dict=result_dict)
        message += ', nCorr: {}'.format(output_dict['corr_scores'].shape[0])
        return message

    def save_pcd(self, points, file_name):
        # 将点数据转换为 Open3D 的 PointCloud 对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # 保存为 PCD 文件
        o3d.io.write_point_cloud(file_name, pcd)
    def after_test_step(self, iteration, data_dict, output_dict, result_dict):
        seq_id = data_dict['seq_id']
        ref_frame = data_dict['ref_frame']
        src_frame = data_dict['src_frame']
        seq_id = str(seq_id)
        
        # 创建保存路径，移除路径中的特殊字符 '@'
        base_path = osp.join(self.output_dir, seq_id)
        if not osp.exists(base_path):
            os.makedirs(base_path)

        # 保存ref_points_f和src_points_f
        np.savetxt(osp.join(base_path, f'geotransformer@{ref_frame}_ref_points_f.csv'), release_cuda(output_dict['ref_points_f']), delimiter=',')
        np.savetxt(osp.join(base_path, f'geotransformer@{src_frame}_src_points_f.csv'), release_cuda(output_dict['src_points_f']), delimiter=',')

        # 保存ref_feats_f和src_feats_f
        np.savetxt(osp.join(base_path, f'geotransformer@{ref_frame}_ref_feats_f.csv'), release_cuda(output_dict['ref_feats_f']), delimiter=',')
        np.savetxt(osp.join(base_path, f'geotransformer@{src_frame}_src_feats_f.csv'), release_cuda(output_dict['src_feats_f']), delimiter=',')

        # 保存匹配文件
        corr_file_name = osp.join(base_path, f'geotransformer@{src_frame}_{ref_frame}_corr.txt')
        with open(corr_file_name, 'w') as f:
            ref_indices = release_cuda(output_dict['ref_indices'])
            src_indices = release_cuda(output_dict['src_indices'])
            for i in range(len(ref_indices)):
                f.write(f'{ref_indices[i]} {src_indices[i]}\n')
        
        # 保存src点云
        src_points = release_cuda(output_dict['src_points'])
        src_pcd_file = osp.join(base_path, f'geotransformer@{src_frame}_src_points.pcd')
        self.save_pcd(src_points, src_pcd_file)

        # 保存ref点云
        ref_points = release_cuda(output_dict['ref_points'])
        ref_pcd_file = osp.join(base_path, f'geotransformer@{ref_frame}_ref_points.pcd')
        self.save_pcd(ref_points, ref_pcd_file)

        # 保存gtmat文件
        gtmat_file_name = osp.join(base_path, f'geotransformer@{src_frame}_{ref_frame}_gtmat.txt')
        with open(gtmat_file_name, 'w') as f:
            transform = release_cuda(data_dict['transform'])
            for i in range(4):
                for j in range(4):
                    f.write(f'{transform[i][j]:.6f}')
                    if j != 3:
                        f.write(' ')
                f.write('\n')

        # 保存标签文件
        label_file_name = osp.join(base_path, f'geotransformer@{src_frame}_{ref_frame}_gt_label.txt')
        with open(label_file_name, 'w') as f:
            ref_corr_points = release_cuda(output_dict['ref_corr_points'])
            corr_scores = release_cuda(output_dict['corr_scores'])
            labels = []

            for i in range(ref_corr_points.shape[0]):
                if corr_scores[i] > 0.05:
                    labels.append(1)
                else:
                    labels.append(0)

def main():
    cfg = make_cfg()
    tester = Tester(cfg)
    tester.run()


if __name__ == '__main__':
    main()

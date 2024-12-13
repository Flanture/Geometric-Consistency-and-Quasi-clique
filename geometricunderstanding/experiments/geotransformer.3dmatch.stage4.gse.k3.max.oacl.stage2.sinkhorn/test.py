import argparse
import os.path as osp
import time

import numpy as np

from geotransformer.engine import SingleTester
from geotransformer.utils.torch import release_cuda
from geotransformer.utils.common import ensure_dir, get_log_string

from dataset import test_data_loader
from config import make_cfg
from model import create_model
from loss import Evaluator


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', choices=['3DMatch', '3DLoMatch', 'val'], help='test benchmark')
    return parser


class Tester(SingleTester):
    def __init__(self, cfg):
        super().__init__(cfg, parser=make_parser())

        # dataloader
        start_time = time.time()
        data_loader, neighbor_limits = test_data_loader(cfg, self.args.benchmark)
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
        self.output_dir = osp.join(cfg.feature_dir, self.args.benchmark)
        ensure_dir(self.output_dir)

    def test_step(self, iteration, data_dict):
        output_dict = self.model(data_dict)
        return output_dict

    def eval_step(self, iteration, data_dict, output_dict):
        result_dict = self.evaluator(output_dict, data_dict)
        return result_dict

    def summary_string(self, iteration, data_dict, output_dict, result_dict):
        scene_name = data_dict['scene_name']
        ref_frame = data_dict['ref_frame']
        src_frame = data_dict['src_frame']
        message = f'{scene_name}, id0: {ref_frame}, id1: {src_frame}'
        message += ', ' + get_log_string(result_dict=result_dict)
        message += ', nCorr: {}'.format(output_dict['corr_scores'].shape[0])
        return message

    def after_test_step(self, iteration, data_dict, output_dict, result_dict):
        # return
        scene_name = data_dict['scene_name']
        ref_id = data_dict['ref_frame']
        src_id = data_dict['src_frame']

        ensure_dir(osp.join(self.output_dir, scene_name))

        base_path = osp.join(self.output_dir, scene_name, f'cloud_bin_{src_id}+cloud_bin_{ref_id}@')

        np.savetxt(f'{base_path}ref_points_f.csv', release_cuda(output_dict['ref_points_f']), delimiter=',')
        np.savetxt(f'{base_path}src_points_f.csv', release_cuda(output_dict['src_points_f']), delimiter=',')
        
        # # 精匹配的高维特征
        np.savetxt(f'{base_path}ref_feats_f.csv', release_cuda(output_dict['ref_feats_f']), delimiter=',') 
        np.savetxt(f'{base_path}src_feats_f.csv', release_cuda(output_dict['src_feats_f']), delimiter=',')       
        
        
        
        corr_file_name = osp.join(self.output_dir, scene_name, f'cloud_bin_{src_id}+cloud_bin_{ref_id}@corr_indices_geotransformer.txt')
        with open(corr_file_name, 'w') as f:
            ref_indices = release_cuda(output_dict['ref_indices'])
            src_indices = release_cuda(output_dict['src_indices'])
            for i in range(len(ref_indices)):
                # f.write(f"{src_corr_points[i, 0]:.6f} {src_corr_points[i, 1]:.6f} {src_corr_points[i, 2]:.6f} "
                #     f"{ref_corr_points[i, 0]:.6f} {ref_corr_points[i, 1]:.6f} {ref_corr_points[i, 2]:.6f}\n")
                f.write(f"{src_indices[i]} {ref_indices[i]}\n")
        # # 2. 输出 GTmat
        gtmat_file_name = osp.join(self.output_dir, scene_name, f'cloud_bin_{src_id}+cloud_bin_{ref_id}@GTmat_geotransformer.txt')
        with open(gtmat_file_name, 'w') as f:
            transform = release_cuda(data_dict['transform'])
            for i in range(4):
                for j in range(4):
                    f.write(f"{transform[i, j]:.6f} ")
                f.write('\n')

        # 3. 输出 label
        label_file_name = osp.join(self.output_dir, scene_name, f'cloud_bin_{src_id}+cloud_bin_{ref_id}@label_geotransformer.txt')
        with open(label_file_name, 'w') as f:
            ref_corr_points = release_cuda(output_dict['ref_corr_points'])
            src_corr_points = release_cuda(output_dict['src_corr_points'])
            gt_node_corr_indices = release_cuda(output_dict['gt_node_corr_indices'])
            corr_scores = release_cuda(output_dict["corr_scores"])

            labels = []
            for i in range(ref_corr_points.shape[0]):
                if corr_scores[i] > 0.05:
                    labels.append(1)  # inlier
                else:
                    labels.append(0)  # outlier
            
            for label in labels:
                f.write(f"{label}\n")
def main():
    cfg = make_cfg()
    tester = Tester(cfg)
    tester.run()


if __name__ == '__main__':
    main()

import torch
import torch.nn as nn
import numpy as np
import os
from geotransformer.modules.loss import WeightedCircleLoss
from geotransformer.modules.ops.transformation import apply_transform
from geotransformer.modules.registration.metrics import isotropic_transform_error
from geotransformer.modules.ops.pairwise_distance import pairwise_distance


class CoarseMatchingLoss(nn.Module):
    def __init__(self, cfg):
        super(CoarseMatchingLoss, self).__init__()
        self.weighted_circle_loss = WeightedCircleLoss(
            cfg.coarse_loss.positive_margin,
            cfg.coarse_loss.negative_margin,
            cfg.coarse_loss.positive_optimal,
            cfg.coarse_loss.negative_optimal,
            cfg.coarse_loss.log_scale,
        )
        self.positive_overlap = cfg.coarse_loss.positive_overlap

    def forward(self, output_dict):
        ref_feats = output_dict['ref_feats_c']
        src_feats = output_dict['src_feats_c']
        gt_node_corr_indices = output_dict['gt_node_corr_indices']
        gt_node_corr_overlaps = output_dict['gt_node_corr_overlaps']
        gt_ref_node_corr_indices = gt_node_corr_indices[:, 0]
        gt_src_node_corr_indices = gt_node_corr_indices[:, 1]

        feat_dists = torch.sqrt(pairwise_distance(ref_feats, src_feats, normalized=True))

        overlaps = torch.zeros_like(feat_dists)
        overlaps[gt_ref_node_corr_indices, gt_src_node_corr_indices] = gt_node_corr_overlaps
        pos_masks = torch.gt(overlaps, self.positive_overlap)
        neg_masks = torch.eq(overlaps, 0)
        pos_scales = torch.sqrt(overlaps * pos_masks.float())

        loss = self.weighted_circle_loss(pos_masks, neg_masks, feat_dists, pos_scales)

        return loss


class FineMatchingLoss(nn.Module):
    def __init__(self, cfg):
        super(FineMatchingLoss, self).__init__()
        self.positive_radius = cfg.fine_loss.positive_radius

    def forward(self, output_dict, data_dict):
        ref_node_corr_knn_points = output_dict['ref_node_corr_knn_points']
        src_node_corr_knn_points = output_dict['src_node_corr_knn_points']
        ref_node_corr_knn_masks = output_dict['ref_node_corr_knn_masks']
        src_node_corr_knn_masks = output_dict['src_node_corr_knn_masks']
        matching_scores = output_dict['matching_scores']
        transform = data_dict['transform']

        src_node_corr_knn_points = apply_transform(src_node_corr_knn_points, transform)
        dists = pairwise_distance(ref_node_corr_knn_points, src_node_corr_knn_points)  # (B, N, M)
        gt_masks = torch.logical_and(ref_node_corr_knn_masks.unsqueeze(2), src_node_corr_knn_masks.unsqueeze(1))
        gt_corr_map = torch.lt(dists, self.positive_radius ** 2)
        gt_corr_map = torch.logical_and(gt_corr_map, gt_masks)
        slack_row_labels = torch.logical_and(torch.eq(gt_corr_map.sum(2), 0), ref_node_corr_knn_masks)
        slack_col_labels = torch.logical_and(torch.eq(gt_corr_map.sum(1), 0), src_node_corr_knn_masks)

        labels = torch.zeros_like(matching_scores, dtype=torch.bool)
        labels[:, :-1, :-1] = gt_corr_map
        labels[:, :-1, -1] = slack_row_labels
        labels[:, -1, :-1] = slack_col_labels

        loss = -matching_scores[labels].mean()

        return loss


class OverallLoss(nn.Module):
    def __init__(self, cfg):
        super(OverallLoss, self).__init__()
        self.coarse_loss = CoarseMatchingLoss(cfg)
        self.fine_loss = FineMatchingLoss(cfg)
        self.weight_coarse_loss = cfg.loss.weight_coarse_loss
        self.weight_fine_loss = cfg.loss.weight_fine_loss

    def forward(self, output_dict, data_dict):
        coarse_loss = self.coarse_loss(output_dict)
        fine_loss = self.fine_loss(output_dict, data_dict)

        loss = self.weight_coarse_loss * coarse_loss + self.weight_fine_loss * fine_loss

        return {
            'loss': loss,
            'c_loss': coarse_loss,
            'f_loss': fine_loss,
        }


class Evaluator(nn.Module):
    def __init__(self, cfg):
        super(Evaluator, self).__init__()
        self.acceptance_overlap = cfg.eval.acceptance_overlap
        self.acceptance_radius = cfg.eval.acceptance_radius
        self.acceptance_rmse = cfg.eval.rmse_threshold
        self.fmr_threshold = cfg.eval.fmr_threshold  # 新增FMR阈值（例如5%）

    @torch.no_grad()
    def evaluate_coarse(self, output_dict):
        ref_length_c = output_dict['ref_points_c'].shape[0]
        src_length_c = output_dict['src_points_c'].shape[0]
        gt_node_corr_overlaps = output_dict['gt_node_corr_overlaps']
        gt_node_corr_indices = output_dict['gt_node_corr_indices']
        masks = torch.gt(gt_node_corr_overlaps, self.acceptance_overlap)
        gt_node_corr_indices = gt_node_corr_indices[masks]
        gt_ref_node_corr_indices = gt_node_corr_indices[:, 0]
        gt_src_node_corr_indices = gt_node_corr_indices[:, 1]
        gt_node_corr_map = torch.zeros(ref_length_c, src_length_c).cuda()
        gt_node_corr_map[gt_ref_node_corr_indices, gt_src_node_corr_indices] = 1.0

        ref_node_corr_indices = output_dict['ref_node_corr_indices']
        src_node_corr_indices = output_dict['src_node_corr_indices']

        precision = gt_node_corr_map[ref_node_corr_indices, src_node_corr_indices].mean()

        return precision

    @torch.no_grad()
    def evaluate_fine(self, output_dict, data_dict):
        transform = data_dict['transform']
        ref_corr_points = output_dict['ref_corr_points']
        src_corr_points = output_dict['src_corr_points']
        src_corr_points = apply_transform(src_corr_points, transform)
        corr_distances = torch.linalg.norm(ref_corr_points - src_corr_points, dim=1)
        precision = torch.lt(corr_distances, self.acceptance_radius).float().mean()
        return precision

    @torch.no_grad()
    def evaluate_registration(self, output_dict, data_dict):
        transform = data_dict['transform']
        est_transform = output_dict['estimated_transform']
        src_points = output_dict['src_points']

        rre, rte = isotropic_transform_error(transform, est_transform)

        realignment_transform = torch.matmul(torch.inverse(transform), est_transform)
        realigned_src_points_f = apply_transform(src_points, realignment_transform)
        rmse = torch.linalg.norm(realigned_src_points_f - src_points, dim=1).mean()
        recall = torch.lt(rmse, self.acceptance_rmse).float()

        return rre, rte, rmse, recall

    @torch.no_grad()
    def evaluate_fmr(self, f_precision):
        """
        计算Feature Matching Recall (FMR)，判断Inlier Ratio超过阈值的点云对比例。
        这里用f_precision来表示Inlier Ratio (IR)。
        """
        fmr = (f_precision > self.fmr_threshold).float().mean()
        return fmr

    def save_matching_results(self, output_dict, data_dict, save_dir):
        """
        将匹配结果保存为 .npz 文件。
        """
        return
        # 将数据从 Tensor 转换为 numpy
        ref_corr_points = output_dict['ref_corr_points'].cpu().numpy()
        src_corr_points = output_dict['src_corr_points'].cpu().numpy()
        corr_scores = output_dict.get('corr_scores', torch.ones(len(ref_corr_points))).cpu().numpy()
        transform = data_dict['transform'].cpu().numpy()
        estimated_transform = output_dict['estimated_transform'].cpu().numpy()

        # 原始点云（如果 data_dict 中包含原始点云）
        ref_points = output_dict['ref_points'].cpu().numpy()  # 假设存在键 'ref_points'
        src_points = output_dict['src_points'].cpu().numpy()  # 假设存在键 'src_points'

        # 将 numpy 数组转换为 torch.Tensor
        ref_corr_points_tensor = torch.tensor(ref_corr_points, device='cuda')
        src_corr_points_tensor = torch.tensor(src_corr_points, device='cuda')
        transform_tensor = torch.tensor(transform, device='cuda')

        # 使用 apply_transform 计算配准后的源点
        src_corr_points_transformed = apply_transform(src_corr_points_tensor, transform_tensor)

        # 计算 inlier_mask
        inlier_mask = torch.lt(
            torch.linalg.norm(ref_corr_points_tensor - src_corr_points_transformed, dim=1),
            self.acceptance_radius
        ).cpu().numpy()  # 转换为 numpy 数组

        # 构建保存文件名
        scene_name = data_dict.get('scene_name', 'unknown_scene')
        ref_frame = data_dict.get('ref_frame', 0)
        src_frame = data_dict.get('src_frame', 0)
        file_name = f'{scene_name}_{ref_frame}_{src_frame}_matching_results.npz'
        save_path = os.path.join(save_dir, file_name)

        # 保存数据到 .npz 文件，包括原始点云
        np.savez(save_path,
                ref_corr_points=ref_corr_points,
                src_corr_points=src_corr_points,
                corr_scores=corr_scores,
                transform=transform,
                estimated_transform=estimated_transform,
                inlier_mask=inlier_mask,
                ref_points=ref_points,  # 原始参考点云
                src_points=src_points)  # 原始源点云
        print(f'Saved matching results to {save_path}')

    def forward(self, output_dict, data_dict, save_results=True, save_dir='./results'):
        c_precision = self.evaluate_coarse(output_dict)
        f_precision = self.evaluate_fine(output_dict, data_dict)
        rre, rte, rmse, recall = self.evaluate_registration(output_dict, data_dict)

        # 使用 f_precision 作为 Inlier Ratio (IR) 来计算 FMR
        fmr = self.evaluate_fmr(f_precision)

        # 保存匹配结果
        if save_results:
            os.makedirs(save_dir, exist_ok=True)
            self.save_matching_results(output_dict, data_dict, save_dir)

        return {
            'PIR': c_precision,
            'IR': f_precision,
            'RRE': rre,
            'RTE': rte,
            'RMSE': rmse,
            'RR': recall,
            'FMR': fmr,
        }
import numpy as np
import torch
import torch.nn as nn
import time
import torch.nn.functional as F

from geotransformer.modules.ops import pairwise_distance
from geotransformer.modules.transformer import SinusoidalPositionalEmbedding, RPEConditionalTransformer

class GeoPPFDescriptor(nn.Module):
    def __init__(self, input_dim, output_dim, k=10):
        super(GeoPPFDescriptor, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.k = k

        # 定义一个简单的Transformer结构
        # self.transformer_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=8)
        # self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=6)

        # 最终的全连接层将Transformer的输出映射到所需的输出维度
        self.fc = nn.Linear(input_dim, output_dim)

    def compute_normals(self, points):
        """
        计算每个点的法向量，使用协方差矩阵方法
        """
        dists = torch.cdist(points, points)  # 计算点云中所有点对的距离
        knn = dists.topk(self.k+1, largest=False)[1][:,1:]  # 找到最近的k个邻居，排除自身

        normals = []
        for i in range(len(points)):
            neighbors = points[knn[i]]  # 获取邻居点
            # 计算协方差矩阵
            mean = neighbors.mean(dim=0)
            cov = (neighbors - mean).T @ (neighbors - mean) / self.k
            # 获取协方差矩阵的特征值和特征向量
            eigvals, eigvecs = torch.linalg.eigh(cov)
            normal = eigvecs[:,0]  # 特征值最小的特征向量作为法向量
            normals.append(normal)
        normals = torch.stack(normals)
        return normals

    def forward(self, points):
        """
        计算整个点云的全局描述符
        """
        # print(points.shape)
        points = points.squeeze(0)
        normals = self.compute_normals(points)
        ppf_features = self.batch_compute_ppf(points, normals)

        # 通过Transformer提取特征
        # transformer_output = self.transformer(ppf_features.view(-1, self.k, 4))
        # transformer_output = transformer_output.view(points.shape[0], points.shape[0], -1)
        # s_x = self.fc(transformer_output.mean(dim=1))  # 平均池化并映射到输出维度
        s_x = self.fc(ppf_features.mean(dim=1))  # 平均池化并映射到输出维度
        
        return s_x.unsqueeze(0)

    def batch_compute_ppf(self, points, normals):
        """
        批量计算PPF特征
        """
        n_points = points.shape[0]
        points_expanded = points.unsqueeze(1).expand(n_points, n_points, 3)
        normals_expanded = normals.unsqueeze(1).expand(n_points, n_points, 3)
        d = points_expanded - points_expanded.transpose(0, 1)
        
        # 计算PPF特征
        norms = torch.norm(d, dim=2)
        angles_ni_d = self.angle_between_batch(normals_expanded, d)
        angles_nj_d = self.angle_between_batch(normals_expanded.transpose(0, 1), d)
        angles_ni_nj = self.angle_between_batch(normals_expanded, normals_expanded.transpose(0, 1))
        
        ppf_features = torch.stack([norms, angles_ni_d, angles_nj_d, angles_ni_nj], dim=2)
        return ppf_features

    def angle_between_batch(self, v1, v2):
        """
        批量计算两个向量之间的角度
        """
        cross_product = torch.cross(v1, v2, dim=2)
        dot_product = torch.sum(v1 * v2, dim=2)
        return torch.atan2(torch.norm(cross_product, dim=2), dot_product)
    
class AttentionFusion(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionFusion, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, 1)

    def forward(self, embeddings):
        embeddings = embeddings.permute(1, 0, 2)
        output = self.attention(embeddings, embeddings, embeddings)
        return output.permute(1, 0, 2)
    
class MambaSelection(nn.Module):
    def __init__(self, embedding_dim):
        super(MambaSelection, self).__init__()
        self.linear = nn.Linear(embedding_dim, 1)
    
    def forward(self, E1, E2):
        g = torch.sigmoid(self.linear(E1))  # Calculate selection weight g
        # print("g:",g)
        E_final = g * E1 + (1 - g) * E2  # Weighted combination
        return E_final


class GeometricStructureEmbedding(nn.Module):
    def __init__(self, hidden_dim, sigma_d, sigma_a, angle_k, reduction_a='max'):
        super(GeometricStructureEmbedding, self).__init__()
        self.sigma_d = sigma_d
        self.sigma_a = sigma_a
        self.factor_a = 180.0 / (self.sigma_a * np.pi)
        self.angle_k = angle_k

        self.embedding = SinusoidalPositionalEmbedding(hidden_dim)
        self.proj_d = nn.Linear(hidden_dim, hidden_dim)
        self.proj_a = nn.Linear(hidden_dim, hidden_dim)

        self.reduction_a = reduction_a
        if self.reduction_a not in ['max', 'mean']:
            raise ValueError(f'Unsupported reduction mode: {self.reduction_a}.')

    @torch.no_grad()
    def get_embedding_indices(self, points):
        r"""Compute the indices of pair-wise distance embedding and triplet-wise angular embedding.

        Args:
            points: torch.Tensor (B, N, 3), input point cloud

        Returns:
            d_indices: torch.FloatTensor (B, N, N), distance embedding indices
            a_indices: torch.FloatTensor (B, N, N, k), angular embedding indices
        """
        batch_size, num_point, _ = points.shape

        dist_map = torch.sqrt(pairwise_distance(points, points))  # (B, N, N)
        d_indices = dist_map / self.sigma_d

        k = self.angle_k
        knn_indices = dist_map.topk(k=k + 1, dim=2, largest=False)[1][:, :, 1:]  # (B, N, k)
        knn_indices = knn_indices.unsqueeze(3).expand(batch_size, num_point, k, 3)  # (B, N, k, 3)
        expanded_points = points.unsqueeze(1).expand(batch_size, num_point, num_point, 3)  # (B, N, N, 3)
        knn_points = torch.gather(expanded_points, dim=2, index=knn_indices)  # (B, N, k, 3)
        ref_vectors = knn_points - points.unsqueeze(2)  # (B, N, k, 3)
        anc_vectors = points.unsqueeze(1) - points.unsqueeze(2)  # (B, N, N, 3)
        ref_vectors = ref_vectors.unsqueeze(2).expand(batch_size, num_point, num_point, k, 3)  # (B, N, N, k, 3)
        anc_vectors = anc_vectors.unsqueeze(3).expand(batch_size, num_point, num_point, k, 3)  # (B, N, N, k, 3)
        sin_values = torch.linalg.norm(torch.cross(ref_vectors, anc_vectors, dim=-1), dim=-1)  # (B, N, N, k)
        cos_values = torch.sum(ref_vectors * anc_vectors, dim=-1)  # (B, N, N, k)
        angles = torch.atan2(sin_values, cos_values)  # (B, N, N, k)
        a_indices = angles * self.factor_a

        return d_indices, a_indices

    def forward(self, points):
        d_indices, a_indices = self.get_embedding_indices(points)

        d_embeddings = self.embedding(d_indices)
        d_embeddings = self.proj_d(d_embeddings)

        a_embeddings = self.embedding(a_indices)
        a_embeddings = self.proj_a(a_embeddings)
        if self.reduction_a == 'max':
            a_embeddings = a_embeddings.max(dim=3)[0]
        else:
            a_embeddings = a_embeddings.mean(dim=3)

        embeddings = d_embeddings + a_embeddings

        return embeddings


class NeighborStructureEmbedding(nn.Module):
    def __init__(self, hidden_dim, sigma_d, scales, sigma_g, k_neighbors):
        super(NeighborStructureEmbedding, self).__init__()
        self.sigma_d = sigma_d
        self.scales = scales
        self.sigma_g = sigma_g
        self.k_neighbors = k_neighbors

        self.rtdie_embedding = nn.Linear(1, hidden_dim)
        self.mdve_embedding = nn.Linear(len(scales) * 2, hidden_dim)
        self.lgee_embedding = nn.Linear(len(scales), hidden_dim)
        self.lgde_embedding = nn.Linear(1, hidden_dim)
        # self.fusion_layer = AttentwionFusion(hidden_dim)

    @torch.no_grad()
    def get_rtdie_indices(self, points):
        # start_time = time.time()
        dist_map = torch.cdist(points, points)  # (B, N, N)
        r_d_indices = torch.exp(-dist_map / self.sigma_d)
        # end_time = time.time()
        # print("get_rtdie_indices time: ", end_time - start_time)
        return r_d_indices

    def get_mdve_indices(self, points): #Multi-Scale Density Variation Embedding indices
        # start_time = time.time()
        B, N, _ = points.shape
        density_indices = []
        gradient_indices = []

        for r in self.scales:
            sphere_volume = (4 / 3) * np.pi * (r ** 3)
            dist_map = torch.sqrt(torch.sum((points.unsqueeze(2) - points.unsqueeze(1)) ** 2, dim=-1))  # (B, N, N)
            within_r = (dist_map < r).float()  # (B, N, N)

            # 计算密度
            density = torch.sum(within_r, dim=-1) / sphere_volume  # (B, N)
            density_indices.append(density)

            # 计算梯度
            density_diff = density.unsqueeze(2) - density.unsqueeze(1)  # (B, N, N)
            squared_diffs = density_diff ** 2
            gradient = torch.sqrt(torch.sum(squared_diffs, dim=-1))  # (B, N)
            gradient_indices.append(gradient)

        density_indices = torch.stack(density_indices, dim=-1)  # (B, N, len(scales))
        gradient_indices = torch.stack(gradient_indices, dim=-1)  # (B, N, len(scales))
        mdve_indices = torch.cat([density_indices, gradient_indices], dim=-1)  # (B, N, 2*len(scales))

        # end_time = time.time()
        # print("get_mdve_indices time: ", end_time - start_time)
        return mdve_indices


    # def get_lgee_indices(self, points): #Local Geometric Entropy Embedding indices
    #     B, N, _ = points.shape
    #     entropy_indices = []

    #     for r in self.scales:
    #         expanded_points = points.unsqueeze(2)  # (B, N, 1, 3)
    #         dists = torch.sqrt(torch.sum((expanded_points - points.unsqueeze(1)) ** 2, dim=-1))  # (B, N, N)
    #         within_r = (dists < r).float()

    #         sector_counts = within_r.sum(dim=-1)  # (B, N)
    #         probabilities = sector_counts / sector_counts.sum(dim=-1, keepdim=True)
    #         entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=-1)
    #         entropy_indices.append(entropy)

    #     entropy_indices = torch.stack(entropy_indices, dim=-1)
    #     return entropy_indices
    

    def get_knn_indices(self, points, k_values=[5,10,25]):
        """
        计算基于最近邻的局部几何熵嵌入 (LGEE) 的索引
        Args:
            points (torch.Tensor): 点云数据，形状为 (B, N, 3)
            k_values (list): 最近邻数量的列表，例如 [k1, k2, k3]
        
        Returns:
            torch.Tensor: 熵索引，形状为 (B, N, len(k_values))
        """
        # start_time = time.time()

        B, N, _ = points.shape
        entropy_indices = []
        adjusted_k_values = [min(k, N - 1) for k in k_values]

        for k in adjusted_k_values:
            # 计算距离矩阵
            expanded_points = points.unsqueeze(2)  # (B, N, 1, 3)
            dists = torch.sqrt(torch.sum((expanded_points - points.unsqueeze(1)) ** 2, dim=-1))  # (B, N, N)
            
            # 找到每个点的k个最近邻点的距离
            knn_dists, _ = torch.topk(dists, k=k + 1, largest=False, sorted=True)  # (B, N, k+1)
            knn_dists = knn_dists[:, :, 1:]  # 去掉自身距离，保留最近邻点距离 (B, N, k)
            
            # 计算每个点的距离熵
            probabilities = knn_dists / knn_dists.sum(dim=-1, keepdim=True)  # (B, N, k)
            entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=-1)  # (B, N)
            entropy_indices.append(entropy)

        entropy_indices = torch.stack(entropy_indices, dim=-1)  # (B, N, len(k_neighbors_list))
        # end_time = time.time()
        # print("get_knn_indices time: ", end_time - start_time)
        return entropy_indices
    

    def get_lgde_indices(self, points): # Local Geometric Diversity Embedding indices 
        # start_time = time.time()
        B, N, _ = points.shape
        # print("points shape: ",{"B":B,"N":N,"_":_})
        # 计算所有点之间的距离
        distances = torch.sqrt(torch.sum((points.unsqueeze(2) - points.unsqueeze(1)) ** 2, dim=-1))  # (B, N, N)
        
        self.k_neighbors = min( int(N * 0.10), self.k_neighbors)
        # 获取每个点的k邻居的索引
        _, indices = distances.topk(self.k_neighbors, dim=-1, largest=False)  # (B, N, k_neighbors)

        # 获取k邻居的点
        batch_indices = torch.arange(B).view(B, 1, 1).expand(B, N, self.k_neighbors)
        knn_points = points[batch_indices, indices]  # (B, N, k_neighbors, 3)

        # 计算中心化后的点云
        mean_centered = knn_points - knn_points.mean(dim=2, keepdim=True)  # (B, N, k_neighbors, 3)

        # 计算协方差矩阵
        cov_matrix = torch.matmul(mean_centered.transpose(-2, -1), mean_centered) / self.k_neighbors  # (B, N, 3, 3)

        # 使用 torch.linalg.eigh 进行对称特征值分解
        eigenvalues, _ = torch.linalg.eigh(cov_matrix)  # (B, N, 3)

        # 特征值按降序排列
        eigenvalues = eigenvalues.flip(dims=[-1])

        # 计算距离方差
        distances_to_neighbors = distances.gather(2, indices)  # (B, N, k_neighbors)
        variances = distances_to_neighbors.var(dim=-1)  # (B, N)

        # 扩展张量以便进行矢量化计算
        sigma_i = variances.unsqueeze(2).expand(B, N, N)  # (B, N, N)
        sigma_j = variances.unsqueeze(1).expand(B, N, N)  # (B, N, N)
        lambda_1_i = eigenvalues[:, :, 0].unsqueeze(2).expand(B, N, N)  # (B, N, N)
        lambda_1_j = eigenvalues[:, :, 0].unsqueeze(1).expand(B, N, N)  # (B, N, N)
        lambda_3_i = eigenvalues[:, :, 2].unsqueeze(2).expand(B, N, N)  # (B, N, N)
        lambda_3_j = eigenvalues[:, :, 2].unsqueeze(1).expand(B, N, N)  # (B, N, N)

        # 计算公式中的各项
        exp_term = torch.exp(-(sigma_i + sigma_j) / (2 * self.sigma_g))
        ratio_term = (lambda_1_i + lambda_1_j) / (lambda_3_i + lambda_3_j)

        # 计算嵌入
        embedding = exp_term * ratio_term  # (B, N, N)

        # 调整形状以适配线性层输入
        embedding = embedding.view(1,B * N, -1)  # (B * N, N)
        # end_time = time.time()
        # print("get_lgde_indices time: ", end_time - start_time)
        return embedding

    def forward(self, points):
        r_d_indices = self.get_rtdie_indices(points)
        mdve_indices = self.get_mdve_indices(points)
        lgee_indices = self.get_knn_indices(points)
        # lgde_indices = self.get_lgde_indices(points)
        # print("r_d_indices:", r_d_indices.shape)
        # print("mdve_indices:", mdve_indices.shape)
        # print("lgee_indices:", lgee_indices.shape)
        # print("lgde_indices:", lgde_indices.shape)
        # 对indices进行标准化
        # r_d_indices = F.normalize(r_d_indices, dim=-1)
        mdve_indices = F.normalize(mdve_indices, dim=-1)
        lgee_indices = F.normalize(lgee_indices, dim=-1)
        # lgde_indices = F.normalize(lgde_indices, dim=-1)

        r_d_embeddings = self.rtdie_embedding(r_d_indices.unsqueeze(-1))
        mdve_embeddings = self.mdve_embedding(mdve_indices)
        lgee_embeddings = self.lgee_embedding(lgee_indices)
        # lgde_embeddings = self.lgde_embedding(lgde_indices.unsqueeze(-1))
        # print("r_d_embeddings:", r_d_embeddings.shape)
        # print("mdve_embeddings:", mdve_embeddings.shape)
        # print("lgee_embeddings:", lgee_embeddings.shape)
        # print("lgde_embeddings:", lgde_embeddings.shape)

        mdve_embeddings = mdve_embeddings.expand_as(r_d_embeddings)
        lgee_embeddings = lgee_embeddings.expand_as(r_d_embeddings)
        # lgde_embeddings = lgde_embeddings.expand_as(r_d_embeddings)
        # 对embeddings进行标准化
        # r_d_embeddings = F.normalize(r_d_embeddings, dim=-1)
        mdve_embeddings = F.normalize(mdve_embeddings, dim=-1)
        lgee_embeddings = F.normalize(lgee_embeddings, dim=-1)
        # lgde_embeddings = F.normalize(lgde_embeddings, dim=-1)

        # embeddings = r_d_embeddings + mdve_embeddings + lgee_embeddings + lgde_embeddings
        # embeddings = r_d_embeddings + mdve_embeddings + lgee_embeddings
        # embeddings = mdve_embeddings
        # embeddings = r_d_embeddings + lgee_embeddings + lgde_embeddings
        embeddings = mdve_embeddings + lgee_embeddings

        return embeddings

class GeometricTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        num_heads,
        blocks,
        sigma_a,
        angle_k,
        sigma_d,
        sigma_g,
        scales,
        k_neighbors,
        dropout=None,
        activation_fn='ReLU',
        reduction_a='max',
    ):
        r"""Geometric Transformer (GeoTransformer).

        Args:
            input_dim: input feature dimension
            output_dim: output feature dimension
            hidden_dim: hidden feature dimension
            num_heads: number of head in transformer
            blocks: list of 'self' or 'cross'
            sigma_d: temperature of distance
            activation_fn: activation function
            reduction_a: reduction mode of angular embedding ['max', 'mean']
        """
        super(GeometricTransformer, self).__init__()

        self.embedding1 = GeometricStructureEmbedding(hidden_dim, sigma_d, sigma_a, angle_k, reduction_a=reduction_a)
        self.embedding2 = NeighborStructureEmbedding(hidden_dim, sigma_d, scales, sigma_g, k_neighbors)
        self.mamba_selection = MambaSelection(hidden_dim)  # Add MambaSelection instance
        # self.global_embedding = GeoPPFDescriptor(4, hidden_dim)
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.transformer = RPEConditionalTransformer(
            blocks, hidden_dim, num_heads, dropout=dropout, activation_fn=activation_fn
        )
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        ref_points,
        src_points,
        ref_feats,
        src_feats,
        ref_masks=None,
        src_masks=None,
    ):
        r"""Geometric Transformer

        Args:
            ref_points (Tensor): (B, N, 3)
            src_points (Tensor): (B, M, 3)
            ref_feats (Tensor): (B, N, C)
            src_feats (Tensor): (B, M, C)
            ref_masks (Optional[BoolTensor]): (B, N)
            src_masks (Optional[BoolTensor]): (B, M)

        Returns:
            ref_feats: torch.Tensor (B, N, C)
            src_feats: torch.Tensor (B, M, C)
        """
        ref_embeddings1 = self.embedding1(ref_points)
        ref_embeddings2 = self.embedding2(ref_points)
        src_embeddings1 = self.embedding1(src_points)
        src_embeddings2 = self.embedding2(src_points)
        # ref_global_embeddings = self.global_embedding(ref_points)
        # src_global_embeddings = self.global_embedding(src_points)
        ref_embeddings = self.mamba_selection(ref_embeddings1, ref_embeddings2)
        src_embeddings = self.mamba_selection(src_embeddings1, src_embeddings2)
        # ref_embeddings = self.mamba_selection(ref_embeddings, ref_global_embeddings)
        # src_embeddings = self.mamba_selection(src_embeddings, src_global_embeddings)

        ref_feats = self.in_proj(ref_feats)
        src_feats = self.in_proj(src_feats)

        ref_feats, src_feats = self.transformer(
            ref_feats,
            src_feats,
            ref_embeddings,
            src_embeddings,
            masks0=ref_masks,
            masks1=src_masks,
        )

        ref_feats = self.out_proj(ref_feats)
        src_feats = self.out_proj(src_feats)

        return ref_feats, src_feats

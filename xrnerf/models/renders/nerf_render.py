# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from torch import nn

from ..builder import RENDERS
from .base import BaseRender


@RENDERS.register_module()
class NerfRender(BaseRender):
    def __init__(self,
                 white_bkgd=False,
                 raw_noise_std=0,
                 rgb_padding=0,
                 density_bias=0,
                 density_activation='relu',
                 **kwarg):
        super().__init__()  # 对于集成了nn.Module的类型，如果有可学习参数，必须加上这个
        self.white_bkgd = white_bkgd
        self.raw_noise_std = raw_noise_std
        self.rgb_padding = rgb_padding
        self.density_bias = density_bias

        if density_activation == 'softplus':  # Density activation.
            self.density_activation = F.softplus
        elif density_activation == 'relu':
            self.density_activation = F.relu
        else:
            raise NotImplementedError

    def get_depth(self, raw, weights, z_vals):
        depth_map = torch.sum(weights * z_vals, -1)

        device = weights.device
        weights_threshold = torch.tensor(15, dtype=torch.float32, device=device).expand(raw[...,3].shape)
        weights_ge = torch.ge(raw[...,3], weights_threshold).to(torch.uint8)
        first_fit_index = torch.argmax(weights_ge, -1)
        dex_depth_map = z_vals[torch.arange(len(first_fit_index)), first_fit_index]
        return depth_map, dex_depth_map

    def get_disp_map(self, weights, z_vals):
        depth_map = torch.sum(weights * z_vals, -1)
        disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map),
                                  depth_map / torch.sum(weights, -1))
        return disp_map

    def get_weights(self, density_delta):
        alpha = 1 - torch.exp(-density_delta)
        weights = alpha * torch.cumprod(
            torch.cat([
                torch.ones(
                    (alpha.shape[0], 1)).to(alpha.device), 1. - alpha + 1e-10
            ], -1), -1)[:, :-1]
        return weights

    def forward(self, data, is_test=False):
        """Transforms model's predictions to semantically meaningful values.

        Args:
            data: inputs
            is_test: is_test
        Returns:
            rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
            disp_map: [num_rays]. Disparity map. Inverse of depth map.
            acc_map: [num_rays]. Sum of weights along each ray.
            weights: [num_rays, num_samples]. Weights assigned to each sampled color.
            depth_map: [num_rays]. Estimated distance to object.
            ret: return values
        """
        raw = data['raw']
        z_vals = data['z_vals']
        # z_vals: [N_rays, N_samples] for nerf or [N_rays, N_samples+1] for mip
        rays_d = data['rays_d']
        raw_noise_std = 0 if is_test else self.raw_noise_std
        device = raw.device

        dists = z_vals[..., 1:] - z_vals[..., :-1]
        if dists.shape[1] != raw.shape[1]:  # if z_val: [N_rays, N_samples]
            dists = torch.cat([
                dists,
                torch.Tensor([1e10]).to(device).expand(dists[..., :1].shape)
            ], -1)  # [N_rays, N_samples]
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
        rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding

        noise = 0.
        if raw_noise_std > 0.:
            noise = torch.randn(raw[..., 3].shape) * raw_noise_std
            noise = noise.to(device)

        density_delta = self.density_activation(raw[..., 3] + noise +
                                                self.density_bias) * dists
        weights = self.get_weights(density_delta)

        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]
        depth_map, dex_depth_map = self.get_depth(raw, weights, z_vals) 
        disp_map = self.get_disp_map(weights, z_vals)
        acc_map = torch.sum(weights, -1)

        if self.white_bkgd:
            rgb_map = rgb_map + (1. - acc_map[..., None])

        ret = {'rgb': rgb_map, 'disp': disp_map, 'acc': acc_map, 'depth': depth_map, 'dexdepth': dex_depth_map}
        data['weights'] = weights  # 放在data里面，给sample函数用

        return data, ret

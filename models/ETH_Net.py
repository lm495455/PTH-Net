import torch
from torch import nn, einsum
import torch.nn.functional as F
from .blocks import (get_sinusoid_encoding, MaskedConv1D, LayerNorm, THR_layer)


class eth_net(nn.Module):  # PTH-Net
    def __init__(self, n_in,  # input feature dimension
                 n_embd,  # embedding dimension (after convolution)
                 sgp_mlp_dim,  # the numnber of dim in SSD
                 max_len,  # max sequence length
                 arch=(2, 2, 1, 1),  # (embedding, stem, branch_1, branch_1)
                 scale_factor=2,  # dowsampling rate for the branch,
                 with_ln=False,  # if to attach layernorm after conv
                 path_pdrop=0.0,  # droput rate for drop path
                 downsample_type='max',  # how to downsample feature in FPN
                 thr_size=None,  # size of local window for mha
                 k=None,  # the K in SSD
                 use_pos=False,  # use absolute position embedding
                 num_classes=7
                 ):
        super(eth_net, self).__init__()
        if thr_size is None:
            thr_size = [3, 1, 3, 3]
        if k is None:
            k = [1, 3, 5]
        self.arch = arch
        self.thr_size = thr_size
        self.max_len = max_len
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.use_pos = use_pos

        # position embedding (1, C, T), rescaled by 1/sqrt(n_embd)
        if self.use_pos:
            pos_embd_4 = get_sinusoid_encoding(int(self.max_len / 4), n_embd) / (n_embd ** 0.5)
            self.register_buffer("pos_embd_4", pos_embd_4, persistent=False)
            pos_embd_8 = get_sinusoid_encoding(int(self.max_len / 2), n_embd) / (n_embd ** 0.5)
            self.register_buffer("pos_embd_8", pos_embd_8, persistent=False)
            pos_embd = get_sinusoid_encoding(int(self.max_len), n_embd) / (n_embd ** 0.5)
            self.register_buffer("pos_embd", pos_embd, persistent=False)

        # embedding network using convs
        self.embd = nn.ModuleList()
        self.embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            if idx == 0:
                in_channels = n_in
            else:
                in_channels = n_embd
            self.embd.append(MaskedConv1D(
                in_channels, n_embd, self.thr_size[0],
                stride=1, padding=self.thr_size[0] // 2, bias=(not with_ln)
            )
            )
            if with_ln:
                self.embd_norm.append(
                    LayerNorm(n_embd)
                )
            else:
                self.embd_norm.append(nn.Identity())
        self.pos_embed = MaskedConv1D(
            n_embd, n_embd, self.thr_size[0],
            stride=1, padding=self.thr_size[0] // 2, groups=n_embd)
        self.pos_embed_norm = LayerNorm(n_embd)
        self.stem = nn.ModuleList()
        for idx in range(arch[1]):
            self.stem.append(
                THR_layer(n_embd, self.thr_size[1], n_hidden=sgp_mlp_dim, k=k))

        # main branch using transformer with pooling
        self.branch_1 = nn.ModuleList()
        for idx in range(arch[2]):
            self.branch_1.append(THR_layer(n_embd, self.thr_size[2], path_pdrop=path_pdrop,
                                           n_hidden=sgp_mlp_dim, k=k))
        self.branch_2 = nn.ModuleList()
        for idx in range(arch[3]):
            self.branch_2.append(THR_layer(n_embd, self.thr_size[3], path_pdrop=path_pdrop,
                                           n_hidden=sgp_mlp_dim, k=k))

        self.head = nn.Linear(n_embd, num_classes) if num_classes > 0 else nn.Identity()

        if self.scale_factor > 1:
            if downsample_type == 'max':
                kernel_size, stride, padding = \
                    self.scale_factor + 1, self.scale_factor, (self.scale_factor + 1) // 2
                self.downsample = nn.MaxPool1d(
                    kernel_size, stride=stride, padding=padding)
                self.stride = stride
            elif downsample_type == 'avg':
                self.downsample = nn.Sequential(nn.AvgPool1d(self.scale_factor, stride=self.scale_factor, padding=0),
                                                nn.Conv1d(n_embd, n_embd, 1, 1, 0))
                self.stride = self.scale_factor
            else:
                raise NotImplementedError("downsample type error")
        else:
            self.downsample = nn.Identity()
            self.stride = 1

        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear/nn.Conv1d bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    def forward_features(self, x_4, x_8, x, mask_4, mask_8, mask, training=True):
        if training:
            for idx in range(len(self.embd)):
                x_4, mask_4 = self.embd[idx](x_4, mask_4)
                x_4 = self.relu(self.embd_norm[idx](x_4))
                x_8, mask_8 = self.embd[idx](x_8, mask_8)
                x_8 = self.relu(self.embd_norm[idx](x_8))
                x, mask = self.embd[idx](x, mask)
                x = self.relu(self.embd_norm[idx](x))

            # 位置编码
            # if self.use_pos:
            #     pos_4, _ = self.pos_embed(x_4, mask_4)
            #     pos_8, _ = self.pos_embed(x_8, mask_8)
            #     pos, _ = self.pos_embed(x, mask)
            #     x_4 = x_4 + pos_4
            #     x_8 = x_8 + pos_8
            #     x = x + pos
            #     x_4 = self.pos_embed_norm(x_4)
            #     x_8 = self.pos_embed_norm(x_8)
            #     x = self.pos_embed_norm(x)

            B_4, C_4, T_4 = x_4.shape
            B_8, C_8, T_8 = x_8.shape
            B, C, T = x.shape
            if self.use_pos and self.training:
                assert T_4 <= self.max_len / 4 and T_8 <= self.max_len / 2 and T <= self.max_len, "Reached max length."
                pe_4 = self.pos_embd_4
                x_4 = x_4 + pe_4[:, :, :T_4] * mask_4.to(x_4.dtype)
                pe_8 = self.pos_embd_8
                x_8 = x_8 + pe_8[:, :, :T_8] * mask_8.to(x_8.dtype)
                pe = self.pos_embd
                x = x + pe[:, :, :T] * mask.to(x.dtype)

            for idx in range(len(self.stem)):
                x_4, x_8, x, mask_4, mask_8, mask = self.stem[idx](x_4, x_8, x, mask_4, mask_8, mask, True)

            if self.arch[2] == 0:
                return x_4, x_8, x, mask_4, mask_8, mask

            x = self.downsample(x)
            out_mask = F.interpolate(
                mask.to(x.dtype),
                size=torch.div(T, self.stride, rounding_mode='trunc'),
                mode='nearest'
            ).detach()
            x_4 = self.downsample(x_4)
            out_mask_4 = F.interpolate(
                mask_4.to(x_4.dtype),
                size=torch.div(T_4, self.stride, rounding_mode='trunc'),
                mode='nearest'
            ).detach()
            x_8 = self.downsample(x_8)
            out_mask_8 = F.interpolate(
                mask_8.to(x_8.dtype),
                size=torch.div(T_8, self.stride, rounding_mode='trunc'),
                mode='nearest'
            ).detach()
            mask_4 = out_mask_4
            mask_8 = out_mask_8
            mask = out_mask
            # main branch with downsampling
            for idx in range(len(self.branch_1)):
                x_4, x_8, x, mask_4, mask_8, mask = self.branch_1[idx](x_4, x_8, x, mask_4, mask_8, mask, True)

            if self.arch[3] == 0:
                return x_4, x_8, x, mask_4, mask_8, mask

            B_4, C_4, T_4 = x_4.shape
            B_8, C_8, T_8 = x_8.shape
            B, C, T = x.shape
            x = self.downsample(x)
            out_mask = F.interpolate(
                mask.to(x.dtype),
                size=torch.div(T, self.stride, rounding_mode='trunc'),
                mode='nearest'
            ).detach()
            x_4 = self.downsample(x_4)
            out_mask_4 = F.interpolate(
                mask_4.to(x_4.dtype),
                size=torch.div(T_4, self.stride, rounding_mode='trunc'),
                mode='nearest'
            ).detach()
            x_8 = self.downsample(x_8)
            out_mask_8 = F.interpolate(
                mask_8.to(x_8.dtype),
                size=torch.div(T_8, self.stride, rounding_mode='trunc'),
                mode='nearest'
            ).detach()
            mask_4 = out_mask_4
            mask_8 = out_mask_8
            mask = out_mask
            # main branch with downsampling
            for idx in range(len(self.branch_2)):
                x_4, x_8, x, mask_4, mask_8, mask = self.branch_2[idx](x_4, x_8, x, mask_4, mask_8, mask, True)

            return x_4, x_8, x, mask_4, mask_8, mask
        else:
            if x_4 is not None:
                x, mask = x_4, mask_4
                for idx in range(len(self.embd)):
                    x, mask = self.embd[idx](x, mask)
                    x = self.relu(self.embd_norm[idx](x))

                # 位置编码
                B, C, T = x.shape
                if self.use_pos:
                    pe = self.pos_embd_4
                    # add pe to x
                    x = x + pe[:, :, :T] * mask.to(x.dtype)
                for idx in range(len(self.stem)):
                    x, mask = self.stem[idx](x, None, None, mask, None, None, False)

                if self.arch[2] == 0:
                    return x, mask

                x = self.downsample(x)
                out_mask = F.interpolate(
                    mask.to(x.dtype),
                    size=torch.div(T, self.stride, rounding_mode='trunc'),
                    mode='nearest'
                ).detach()
                mask = out_mask
                # main branch with downsampling
                for idx in range(len(self.branch_1)):
                    x, mask = self.branch_1[idx](x, None, None, mask, None, None, False)

                if self.arch[3] == 0:
                    return x, mask

                B, C, T = x.shape
                x = self.downsample(x)
                out_mask = F.interpolate(
                    mask.to(x.dtype),
                    size=torch.div(T, self.stride, rounding_mode='trunc'),
                    mode='nearest'
                ).detach()
                mask = out_mask
                # main branch with downsampling
                for idx in range(len(self.branch_2)):
                    x, mask = self.branch_2[idx](x, None, None, mask, None, None, False)
            elif x_8 is not None:
                x, mask = x_8, mask_8
                for idx in range(len(self.embd)):
                    x, mask = self.embd[idx](x, mask)
                    x = self.relu(self.embd_norm[idx](x))

                # 位置编码
                B, C, T = x.shape
                if self.use_pos:
                    pe = self.pos_embd_8
                    # add pe to x
                    x = x + pe[:, :, :T] * mask.to(x.dtype)

                for idx in range(len(self.stem)):
                    x, mask = self.stem[idx](None, x, None, None, mask, None, False)

                if self.arch[2] == 0:
                    return x, mask

                x = self.downsample(x)
                out_mask = F.interpolate(
                    mask.to(x.dtype),
                    size=torch.div(T, self.stride, rounding_mode='trunc'),
                    mode='nearest'
                ).detach()
                mask = out_mask
                # main branch with downsampling
                for idx in range(len(self.branch_1)):
                    x, mask = self.branch_1[idx](None, x, None, None, mask, None, False)

                if self.arch[3] == 0:
                    return x, mask

                B, C, T = x.shape
                x = self.downsample(x)
                out_mask = F.interpolate(
                    mask.to(x.dtype),
                    size=torch.div(T, self.stride, rounding_mode='trunc'),
                    mode='nearest'
                ).detach()
                mask = out_mask
                # main branch with downsampling
                for idx in range(len(self.branch_2)):
                    x, mask = self.branch_2[idx](None, x, None, None, mask, None, False)
            else:
                for idx in range(len(self.embd)):
                    x, mask = self.embd[idx](x, mask)
                    x = self.relu(self.embd_norm[idx](x))

                # 位置编码
                B, C, T = x.shape
                if self.use_pos:
                    pe = self.pos_embd
                    # add pe to x
                    x = x + pe[:, :, :T] * mask.to(x.dtype)

                for idx in range(len(self.stem)):
                    x, mask = self.stem[idx](None, None, x, None, None, mask, False)

                if self.arch[2] == 0:
                    return x, mask

                x = self.downsample(x)
                out_mask = F.interpolate(
                    mask.to(x.dtype),
                    size=torch.div(T, self.stride, rounding_mode='trunc'),
                    mode='nearest'
                ).detach()
                mask = out_mask
                # main branch with downsampling
                for idx in range(len(self.branch_1)):
                    x, mask = self.branch_1[idx](None, None, x, None, None, mask, False)

                if self.arch[3] == 0:
                    return x, mask

                B, C, T = x.shape
                x = self.downsample(x)
                out_mask = F.interpolate(
                    mask.to(x.dtype),
                    size=torch.div(T, self.stride, rounding_mode='trunc'),
                    mode='nearest'
                ).detach()
                mask = out_mask

                for idx in range(len(self.branch_2)):
                    x, mask = self.branch_2[idx](None, None, x, None, None, mask, False)

            return x, mask

    def forward(self, x_4, x_8, x, mask_4, mask_8, mask, training=True):
        if training:
            x_4, x_8, x, mask_4, mask_8, mask = self.forward_features(x_4, x_8, x, mask_4, mask_8, mask, True)
            # 对x_4, x_8, x按照最后一维取平均值，同时考虑mask_4, mask_8, mask
            x_4 = x_4 * mask_4
            x_8 = x_8 * mask_8
            x = x * mask
            x_4 = x_4.mean(dim=2)
            x_8 = x_8.mean(dim=2)
            x = x.mean(dim=2)
            # x_4 = x_4.max(dim=2).values
            # x_8 = x_8.max(dim=2).values
            # x = x.max(dim=2).values
            x_4 = self.head(x_4)
            x_8 = self.head(x_8)
            x = self.head(x)
            return x_4, x_8, x

            # Ablation
            # x, mask = self.forward_features(None, None, x, None, None, mask, False)
            # x = x * mask
            # x = x.mean(dim=2)
            # # x = x.max(dim=2).values
            # x = self.head(x)
            # return x_4, x_8, x
        else:
            if x_4 is not None:
                x, mask = self.forward_features(x_4, None, None, mask_4, None, None, False)
                x = x * mask
                x = x.mean(dim=2)
                # x = x.max(dim=2).values
                x = self.head(x)
            elif x_8 is not None:
                x, mask = self.forward_features(None, x_8, None, None, mask_8, None, False)
                x = x * mask
                x = x.mean(dim=2)
                # x = x.max(dim=2).values
                x = self.head(x)
            else:
                x, mask = self.forward_features(None, None, x, None, None, mask, False)
                x = x * mask
                x = x.mean(dim=2)
                # x = x.max(dim=2).values
                x = self.head(x)
            return x


def PTH_Net():
    max_len = 16
    k = [1, 3, 5]
    thr_size = [3, 1, 3, 3]
    arch = (2, 2, 1, 1)
    return eth_net(n_in=1408, n_embd=512, mlp_dim=768, max_len=max_len, arch=arch,
                   scale_factor=2, with_ln=True, path_pdrop=0.1, downsample_type='max',
                   thr_size=thr_size,
                   k=k, init_conv_vars=0, use_pos=False)


if __name__ == '__main__':
    img = torch.randn((1, 16, 3, 112, 112))
    model = PTH_Net()
    model(img)

"""
This is an implementation of the VQVAE model used in:
    1.  Khazatsky, Alexander et al. “What Can I Do Here? Learning New Skills by Imagining Visual Affordances.” 2021 IEEE International Conference on Robotics and Automation (ICRA) (2021): 14291-14297.
        https://arxiv.org/abs/2106.00671 
VQVAE:
    2.  "Neural Discrete Representation Learning", van den Oord et al. 2017
        https://arxiv.org/abs/1711.00937

Source: https://github.com/anair13/rlkit/blob/master/rlkit/torch/vae/vq_vae.py
"""

from __future__ import print_function
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
import pytorch_util as ptu
import torch.nn as nn
import torch.nn.functional as F

class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(

            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                out_channels=num_residual_hiddens,
                kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                out_channels=num_hiddens,
                kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers,
            num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList(
            [Residual(in_channels, num_hiddens, num_residual_hiddens)
             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers,
            num_residual_hiddens):
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
            out_channels=num_hiddens // 2,
            kernel_size=4,
            stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens // 2,
            out_channels=num_hiddens,
            kernel_size=4,
            stride=2, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs, ):
        x = self._conv_1(inputs)
        x = F.relu(x)

        x = self._conv_2(x)
        x = F.relu(x)

        x = self._conv_3(x)
        return self._residual_stack(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers,
            num_residual_hiddens, out_channels=3):
        super(Decoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
            out_channels=num_hiddens,
            kernel_size=3,
            stride=1, padding=1)

        self._residual_stack = ResidualStack(in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens)

        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens,
            out_channels=num_hiddens // 2,
            kernel_size=4,
            stride=2, padding=1)

        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens // 2,
            out_channels=out_channels,
            kernel_size=4,
            stride=2, padding=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)

        x = self._residual_stack(x)

        x = self._conv_trans_1(x)
        x = F.relu(x)

        x = self._conv_trans_2(x)

        return torch.clamp(x, 0., 1.)


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost,
            gaussion_prior=False):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._embedding = nn.Embedding(
            self._num_embeddings, self._embedding_dim)

        if gaussion_prior:
            self._embedding.weight.data.normal_()

        else:
            self._embedding.weight.data.uniform_(
                -1 / self._num_embeddings, 1 / self._num_embeddings)

        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        # print("flat_input shape:", flat_input.shape)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings,
            device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs *
                                          torch.log(avg_probs + 1e-10)))

        
        return loss, quantized.permute(0, 3, 1,2).contiguous(), perplexity, encoding_indices


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay,
            epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(
            self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(
            num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings,
            device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(
            encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(
                self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(
                self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        # e_latent_loss = F.mse_loss(quantized.detach(), inputs, reduction='sum')
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs *
                                          torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1,
            2).contiguous(), perplexity, encoding_indices


class VQ_VAE(nn.Module):
    def __init__(
            self,
            embedding_dim=5,
            input_channels=3,
            output_channels=3,
            num_hiddens=128,
            num_residual_layers=3,
            num_residual_hiddens=64,
            num_embeddings=512,
            commitment_cost=0.25,
            recon_weight=0.1,
            entropy_weight=0.01,
            imsize=48,
            decay=0.0,
            ignore_background=False):
        super(VQ_VAE, self).__init__()
        self.imsize = imsize
        self.embedding_dim = embedding_dim
        self.pixel_cnn = None
        self.input_channels = input_channels
        self.imlength = imsize * imsize * input_channels
        self.num_embeddings = num_embeddings

        self._encoder = Encoder(input_channels, num_hiddens,
            num_residual_layers,
            num_residual_hiddens)

        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
            out_channels=self.embedding_dim,
            kernel_size=1,
            stride=1)
        
        self.commitment_cost = commitment_cost
        self.recon_weight = recon_weight
        self.entropy_weight = entropy_weight
        self.ignore_background = ignore_background

        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings,
                self.embedding_dim,
                commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, self.embedding_dim,
                commitment_cost)

        self._decoder = Decoder(self.embedding_dim,
            num_hiddens,
            num_residual_layers,
            num_residual_hiddens, 
            output_channels)

        # Calculate latent sizes
        if imsize == 32:
            self.root_len = 8
        elif imsize == 36:
            self.root_len = 9
        elif imsize == 48:
            self.root_len = 12
        elif imsize == 64:
            self.root_len = 16
        elif imsize == 84:
            self.root_len = 21
        elif imsize == 8:
            self.root_len = 2
        else:
            raise ValueError(imsize)

        self.discrete_size = self.root_len * self.root_len
        self.representation_size = self.discrete_size * self.embedding_dim
        # Calculate latent sizes

    def compute_loss(self, inputs):
        inputs = inputs.view(-1,
            self.input_channels,
            self.imsize,
            self.imsize)

        vq_loss, quantized, perplexity, encoding_indices = self.quantize_image(inputs)
        recon = self.decode(quantized)

        if self.ignore_background:
            # Weight non-zero pixels more than black pixels; for dot/block datasets
            weights = (inputs > 0).float() * 1 + 1  
            recon_loss = F.mse_loss(recon * weights, inputs * weights) * self.recon_weight
        else:
            recon_loss = F.mse_loss(recon, inputs) * self.recon_weight

        outputs = {
            'reconstructions': recon,
            'quantized': quantized,
            'encoding_indices': encoding_indices,
            'perplexity': perplexity
        }

        losses = {
            'vq_loss': vq_loss,
            'recon_loss': recon_loss,
        }

        return outputs, losses

    def codebook_entropy(self, inputs, encoding_indices):
        """ 
        Encourage the model to use more of the embeddings in the codebook 
        by calculating the squared normalized entropy of the codebook usage.
        This term penalizes low entropy (i.e., using only a few embeddings)
        more than it forces high entropy (uniform usage of all embeddings).
        """
        # Codebook usage probabilities
        avg_probs = torch.zeros(self.num_embeddings, device=inputs.device)
        unique_indices, counts = torch.unique(encoding_indices, return_counts=True)
        avg_probs[unique_indices] = counts.float()
        avg_probs = avg_probs / torch.sum(avg_probs)

        # Normalized squared entropy
        entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        max_entropy = np.log(self.num_embeddings)
        entropy_loss = (1 - (entropy / max_entropy))**2 * self.entropy_weight
        return entropy_loss

    def quantize_image(self, inputs):
        inputs = inputs.view(-1,
            self.input_channels,
            self.imsize,
            self.imsize)

        z = self._encoder(inputs)
        z = self._pre_vq_conv(z)
        if self.training:
            z = z + 0.1 * torch.randn_like(z)
        return self._vq_vae(z)

    def encode(self, inputs, cont=True):
        _, quantized, _, encodings = self.quantize_image(inputs)

        if cont:
            return quantized.reshape(-1, self.representation_size)
        return encodings.reshape(-1, self.discrete_size)

    def latent_to_square(self, latents):
        return latents.reshape(-1, self.root_len, self.root_len)

    def discrete_to_cont(self, e_indices):
        e_indices = self.latent_to_square(e_indices)
        input_shape = e_indices.shape + (self.embedding_dim,)
        e_indices = e_indices.reshape(-1).unsqueeze(1)

        min_encodings = torch.zeros(e_indices.shape[0], self.num_embeddings,
            device=e_indices.device)
        min_encodings.scatter_(1, e_indices, 1)

        e_weights = self._vq_vae._embedding.weight

        z_q = torch.matmul(min_encodings, e_weights).view(input_shape)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q

    def set_pixel_cnn(self, pixel_cnn):
        self.pixel_cnn = pixel_cnn

    def sample_conditional_indices(self, batch_size, cond):
        if cond.shape[0] == 1:
            cond = cond.repeat(batch_size, axis=0)
        cond = ptu.from_numpy(cond)

        sampled_indices = self.pixel_cnn.generate(
            shape=(self.root_len, self.root_len),
            batch_size=batch_size,
            cond=cond)

        return sampled_indices

    def sample_prior(self, batch_size, cond=None):
        if self.pixel_cnn.is_conditional:
            sampled_indices = self.sample_conditional_indices(batch_size, cond)
        else:
            sampled_indices = self.pixel_cnn.generate(
                shape=(self.root_len, self.root_len),
                batch_size=batch_size)

        sampled_indices = sampled_indices.reshape(batch_size, self.discrete_size)
        z_q = self.discrete_to_cont(sampled_indices).reshape(-1, self.representation_size)
        return ptu.get_numpy(z_q)

    def decode(self, latents, cont=True):
        if cont:
            z_q = latents.reshape(-1, self.embedding_dim, self.root_len,
                self.root_len)
        else:
            z_q = self.discrete_to_cont(latents)

        return self._decoder(z_q)

    def encode_one_np(self, inputs, cont=True):
        return ptu.get_numpy(self.encode(ptu.from_numpy(inputs), cont=cont))[0]

    def encode_np(self, inputs, cont=True):
        return ptu.get_numpy(self.encode(ptu.from_numpy(inputs), cont=cont))

    def decode_one_np(self, inputs, cont=True):
        return np.clip(ptu.get_numpy(
            self.decode(ptu.from_numpy(inputs).reshape(1, -1), cont=cont))[0],
            0, 1)

    def decode_np(self, inputs, cont=True):
        return np.clip(
            ptu.get_numpy(self.decode(ptu.from_numpy(inputs), cont=cont)), 0, 1)

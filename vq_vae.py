from typing import List, Any

import torch
from torch import embedding, nn, square
from torch.nn import functional as F

class VectorQuantizer(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 ) -> None:
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim  = embedding_dim

        self.embedding = nn.Embedding(self.num_embeddings,
                                      self.embedding_dim
                                      )
        self.embedding.weight.data.uniform_(-0.5, 0.5) #(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, x: torch.Tensor):
        # Convert inputs for BCHW to BHWC
        x = x.permute(0, 2, 3, 1).contiguous()
        shape = x.shape

        # Flatten input
        x_flat = x.view(-1, self.embedding_dim)

        # Calculate distances
        # What this does: For each of the N embedding_dim vectors in x_flat,
        # we subtract their values from each of the num_embeddings embedding_dim in our embedding table, square the result and sum them up
        # distances will be of shape (N, num_embeddings) where
        # distances[0, 0] = (x_flat[0] - self.embedding.weight[0]).square().sum()
        # and distances[0, 1] = (x_flat[0] - self.embedding.weight[1]).square().sum()
        distances = (torch.sum(
            x_flat.square(), dim=1, keepdim=True
        ) + 
                     torch.sum(self.embedding.weight.square(), dim=1)

        - 2 * x_flat @ self.embedding.weight.t()
                               )

        # Encoding
        # This produces a tensor of shape (BHW, 1).
        # It goes along each row in distances and picks the index with the smallest value
        encoding_indices = torch.argmin(distances, dim=1)

        
        q = self.embedding.weight[encoding_indices].view(shape)

        # Loss calculation
        # Loss = log P(x|z_{q}(x)) + ||z_{e}(x).detach() - e|| + Beta * ||z_{e}(x) - e.detach()||
        # where || = MSE
        commitment_loss = F.mse_loss(x, q.detach(),
                                     reduction='none').sum(dim=[1, 2, 3]).mean()
        quantization_loss = F.mse_loss(x.detach(), q,
                                       reduction='none').sum(dim=[1, 2, 3]).mean()

        #import code; code.interact(local=locals())

        q = x + (q - x).detach() # Straight through estimator
        q = q.permute(0, 3, 1, 2).contiguous()
        counts = torch.bincount(encoding_indices,
                                   minlength=self.num_embeddings).float()
        avg_probs = counts / counts.sum()

        # Only sum over nonzero probs
        nonzero_probs = avg_probs[avg_probs > 0]
        perplexity = torch.exp(-torch.sum(nonzero_probs * torch.log(nonzero_probs + 1e-8)))
        return quantization_loss, commitment_loss, q, perplexity

class Conv(nn.Module):
    def __init__(self,
                 in_channels: int = 32,
                 out_channels: int = 64,
                 stride: int = 1
                 ):
        super().__init__()

        self.proj = None
        if stride == 2 or in_channels != out_channels:
            self.proj = nn.Conv2d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  stride=stride,
                                  kernel_size=1,
                                  )
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=in_channels // 4,
                               kernel_size=1,
                               #padding=1,
                               )
        self.bn1   = nn.BatchNorm2d(in_channels // 4)
        self.conv2 = nn.Conv2d(in_channels=in_channels // 4,
                               out_channels=in_channels // 4,
                               kernel_size=3,
                               padding=1,
                               stride=stride)
        self.bn2    = nn.BatchNorm2d(in_channels // 4)

        self.conv3 = nn.Conv2d(in_channels=in_channels // 4,
                               out_channels=out_channels,
                               kernel_size=1,
                               #padding=1,
                               stride=1)
        self.bn3  = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)

        identity = identity if self.proj is None else self.proj(identity)
        x = identity + x
        x = F.leaky_relu(self.bn3(x))
        return x

class ConvBlock(nn.Module):
    def __init__(self, 
                 in_channels: int = 32,
                 out_channels: int = 64,
                 stride: int = 1,
                 n_repeats: int = 3
                 ) -> None:
        super().__init__()

        layers = []
        for i in range(n_repeats - 1):
            layers.append(Conv(in_channels=in_channels,
                               out_channels=in_channels,
                               stride=1)
                          )

        layers.append(Conv(in_channels=in_channels,
                           out_channels=out_channels,
                           stride=stride)
                      )

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

class DeconvBlock(nn.Module):
    def __init__(self, 
                 in_channels: int = 32,
                 out_channels: int = 64,
                 stride: int = 1,
                 n_repeats: int = 3,
                 interpolate: bool = True
                 ) -> None:
        super().__init__()
        self.interpolate = interpolate
        self.conv = ConvBlock(in_channels,
                              out_channels,
                              stride,
                              n_repeats)

    def forward(self, x):
        if self.interpolate:
            x = F.interpolate(x, scale_factor=2)
        return self.conv(x)

class Encoder(nn.Module):
    def __init__(self, channels: List[int] = [32, 64, 128],
                 n_repeats: List[int] = [3, 3, 4],
                 embedding_dim: int = 10
                 ) -> None:
        super().__init__()
        
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels=3,
                                out_channels=channels[0],
                                stride=1,
                                kernel_size=3,
                                padding=1),
                                    nn.BatchNorm2d(channels[0]),
                                    nn.LeakyReLU()
                                )
        self.conv1 = ConvBlock(in_channels=channels[0],
                               out_channels=channels[1],
                               stride=2,
                               n_repeats=n_repeats[0])

        self.conv2 = ConvBlock(in_channels=channels[1],
                               out_channels=channels[2],
                               stride=2,
                               n_repeats=n_repeats[1]
                               )

        self.conv3 = ConvBlock(in_channels=channels[2],
                               out_channels=embedding_dim,
                               stride=1,
                               n_repeats=n_repeats[2])

    def forward(self, x):
        # x.shape: (B, 3, 32, 32) for CIFAR10
        x = self.layer1(x)
        x = self.conv1(x) # (B, C, 32. 32) => (B. C, 16, 16)
        x = self.conv2(x) # (B, C, 16, 16) => (B, C. 16, 16)
        x = self.conv3(x) # (B, C, 16, 16) => (B, C, 8, 8)
        #x = self.conv4(x) # (B, C, 8, 8)   => (B, C, 8, 8)
        return x

class Decoder(nn.Module):
    def __init__(self, channels: List[int] = [128, 64, 32],
                 n_repeats: List[int] = [3, 3, 4],
                 embedding_dim: int = 10
                 ) -> None:
        super().__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(in_channels=embedding_dim,
                                out_channels=channels[0],
                                stride=1,
                                kernel_size=3,
                                padding=1),
                                    nn.BatchNorm2d(channels[0]),
                                    nn.LeakyReLU()
                                )
        self.conv1 = DeconvBlock(in_channels=channels[0],
                               out_channels=channels[1],
                               stride=1,
                               n_repeats=n_repeats[0])

        self.conv2 = DeconvBlock(in_channels=channels[1],
                               out_channels=channels[2],
                               stride=1,
                               n_repeats=n_repeats[1],
                            )

        self.conv3 = DeconvBlock(in_channels=channels[2],
                               out_channels=3,
                               stride=1,
                               n_repeats=n_repeats[2], interpolate=False)
        self.conv4 = nn.Conv2d(in_channels=3,
                               out_channels=3,
                               kernel_size=3,
                               padding=1)

    def forward(self, q):
        # q shape: (B, 10, 8, 8)
        x = self.layer1(q) # (B, 10, 8, 8) => (B, 128, 8, 8)
        x = self.conv1(x) # (B, 128, 8, 8) => (B, 64, 16, 16)
        x = self.conv2(x) # (B, 64, 16, 16) => (B, 32, 32, 32)
        x = self.conv3(x) # (B, 32, 32, 32) => (B, 3, 32, 32)
        x = self.conv4(x)
        return F.sigmoid(x)

class VQVAE(nn.Module):
    def __init__(self,
                 encoder = Encoder,
                 decoder = Decoder,
                 vector_q = VectorQuantizer,
                 channels: List[int] = [64, 128, 256],
                 n_repeats: List[int] = [3, 4, 4],
                 embedding_dim: int = 10,
                 num_embeddings: int = 256,
                 ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim  = embedding_dim

        self.encoder = encoder(channels=channels,
                               n_repeats=n_repeats,
                               embedding_dim=embedding_dim)
        self.decoder = decoder(channels=channels[::-1],
                               n_repeats=n_repeats,
                               embedding_dim=embedding_dim)
        self.vector_q= vector_q(embedding_dim=embedding_dim, 
                                num_embeddings=num_embeddings,
                                )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        #import code; code.interact(local=locals())
        q, q_loss, c_loss, perplexity = self.encode(x)
        x_recon = self.decoder(q)
        return x_recon, q_loss, c_loss, perplexity

    def encode(self, x):
        encoder_output = self.encoder(x)
        quantization_loss, commitment_loss, q, perplexity = self.vector_q(encoder_output)
        return q, quantization_loss, commitment_loss, perplexity

    def decode(self, q):
        return self.decoder(q)

if __name__ == "__main__":
    vq = VectorQuantizer(32, 10)
    enc = Encoder()
    dec = Decoder()
    vae = VQVAE()
    print(sum([p.numel() for p in enc.parameters()]))
    print(sum([p.numel() for p in dec.parameters()]))
    print(sum([p.numel() for p in vae.parameters()]))

    x = torch.rand(8, 3, 32, 32)
    print(x.min(), x.max(), x.mean())
    out = enc(x)
    x2 = dec(out)
    x3, qloss, closs, perp2 = vae(x)
    qloss, closs, q, perp = vq(out)
    print(qloss, perp, q.shape, x.shape, out.shape, x2.shape, x3.shape)



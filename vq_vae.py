import torch
from torch import embedding, nn, square
from torch.nn import functional as F

class VectorQuantizer(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 commitment_cost: float) -> None:
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim  = embedding_dim
        self.commitment_cost= commitment_cost

        self.embedding = nn.Embedding(self.num_embeddings,
                                      self.embedding_dim
                                      )
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

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
        commitment_loss = F.mse_loss(x, q.detach())
        quantization_loss = F.mse_loss(x.detach(), q)
        loss = quantization_loss + self.commitment_cost * commitment_loss

        q = x + (q - x).detach() # Straight through estimator
        q = q.permute(0, 3, 1, 2).contiguous()
        counts = torch.bincount(encoding_indices,
                                   minlength=self.num_embeddings).float()
        avg_probs = counts / counts.sum()

        # Only sum over nonzero probs
        nonzero_probs = avg_probs[avg_probs > 0]
        perplexity = torch.exp(-torch.sum(nonzero_probs * torch.log(nonzero_probs + 1e-8)))
        return loss, q, perplexity

if __name__ == "__main__":
    vq = VectorQuantizer(32, 10, 0.25)

    x = torch.randn(8, 10, 4, 4)
    loss, q, perp = vq(x)
    print(loss, perp, q.shape)


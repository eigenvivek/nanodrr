import torch


class Transform(torch.nn.Module):
    """Wraps a homogeneous (N+1)x(N+1) matrix as a callable transform."""

    def __init__(self, matrix):
        super().__init__()
        if matrix.ndim == 2:
            matrix = matrix.unsqueeze(0)
        self.register_buffer("matrix", matrix)

    @staticmethod
    def _homo(v):
        return torch.cat([v, torch.ones_like(v[..., :1])], dim=-1)

    @staticmethod
    def _dehomo(v):
        return v[..., :-1] / v[..., -1:]

    @property
    def n(self):
        return self.matrix.shape[-1] - 1

    def forward(self, v, direction=False):
        """Apply transform to points (..., N) or direction vectors."""
        M = self.matrix
        if direction:
            return torch.einsum("...ij,...j->...i", M[..., : self.n, : self.n], v)
        return self._dehomo(torch.einsum("...ij,...j->...i", M, self._homo(v)))

    def inverse(self):
        return Transform(torch.linalg.inv(self.matrix))

    def compose(self, other):
        """self then other."""
        return Transform(torch.einsum("...ij,...jk->...ik", other.matrix, self.matrix))

    def __matmul__(self, other):
        """T1 @ T2: apply T2 first, then T1."""
        return Transform(torch.einsum("...ij,...jk->...ik", self.matrix, other.matrix))

    def __len__(self):
        return len(self.matrix)

    def __repr__(self):
        return f"Transform(n={self.n}, batch={self.matrix.shape[:-2]})"

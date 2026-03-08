import torch

from nanodrr.geometry.transform import transform_point


def test_transform_point_identity():
    """Identity transform should leave points unchanged."""
    B = 2
    N = 3
    xform = torch.eye(N + 1).expand(B, N + 1, N + 1).clone()
    pts = torch.randn(B, 5, N)

    out = transform_point(xform, pts)
    torch.testing.assert_close(out, pts)


def test_transform_point_translation_only():
    """Pure translation should shift all points by the same amount."""
    N = 3
    translation = torch.tensor([1.0, -2.0, 0.5])
    xform = torch.eye(N + 1)
    xform[:N, 3] = translation
    xform = xform.unsqueeze(0)  # (1, 4, 4)

    pts = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
        ],
        dtype=torch.float32,
    ).unsqueeze(0)  # (1, 2, 3)

    out = transform_point(xform, pts)
    expected = pts + translation
    torch.testing.assert_close(out, expected)


def test_transform_point_composition():
    """Sequential transforms should match composition of matrices."""
    B = 1
    N = 3

    # Two arbitrary affine transforms
    A = torch.eye(N + 1)
    A[:N, :N] = torch.tensor(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    A[:N, 3] = torch.tensor([1.0, 0.0, 0.0])

    Bmat = torch.eye(N + 1)
    Bmat[:N, 3] = torch.tensor([0.0, 2.0, -1.0])

    A = A.unsqueeze(0)  # (1, 4, 4)
    Bmat = Bmat.unsqueeze(0)

    pts = torch.randn(B, 10, N)

    out_seq = transform_point(Bmat, transform_point(A, pts))
    out_comp = transform_point(Bmat @ A, pts)
    torch.testing.assert_close(out_seq, out_comp)


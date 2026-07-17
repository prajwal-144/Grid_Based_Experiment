import torch

from physics_losses import (
    SourcePlaneIntensityConsistency,
    build_fixed_sis_information_maps,
    fixed_sis_magnification,
    magnification_adaptive_loss,
    magnification_adaptive_source_grid,
)


def test_fixed_sis_magnification_is_finite_and_clipped():
    signed, absolute = fixed_sis_magnification(
        image_shape=32,
        pixel_scale_arcsec=0.05,
        theta_e_arcsec=0.5,
        mu_clip=12.0,
    )
    assert signed.shape == (1, 1, 32, 32)
    assert torch.isfinite(signed).all()
    assert torch.isfinite(absolute).all()
    assert absolute.max() <= 12.0 + 1e-6


def test_information_map_identity_mapping_preserves_shape():
    size = 8
    indices = torch.arange(size * size).repeat(2, 1)
    values = torch.ones(size * size)
    mapping = torch.sparse_coo_tensor(
        indices,
        values,
        (size * size, size * size),
    ).coalesce()
    maps = build_fixed_sis_information_maps(
        backward_mapping=mapping,
        image_shape=size,
        pixel_scale_arcsec=0.1,
        theta_e_arcsec=0.3,
        mu_clip=10.0,
    )
    assert maps.source_information_lr.shape == (1, 1, size, size)
    assert torch.allclose(maps.source_information_lr, maps.image_information)
    assert (maps.source_information_lr >= 0).all()
    assert (maps.source_information_lr <= 1).all()


def test_adaptive_grid_preserves_constant_surface_brightness():
    source = torch.ones(2, 1, 16, 16)
    info = torch.rand(1, 1, 16, 16)
    adaptive, _ = magnification_adaptive_source_grid(source, info)
    assert torch.allclose(adaptive, source, atol=1e-6)
    loss = magnification_adaptive_loss(source, adaptive, info)
    assert loss.item() < 1e-8


def test_adaptive_grid_uses_fine_and_coarse_limits():
    source = torch.zeros(1, 1, 16, 16)
    source[:, :, 8, 8] = 1.0
    info_high = torch.ones(1, 1, 16, 16)
    info_low = torch.zeros(1, 1, 16, 16)
    adaptive_high, _ = magnification_adaptive_source_grid(source, info_high)
    adaptive_low, diagnostics = magnification_adaptive_source_grid(source, info_low)
    assert torch.allclose(adaptive_high, source, atol=1e-6)
    assert torch.allclose(adaptive_low, diagnostics["coarse_source"], atol=1e-6)


def test_source_consistency_detects_disagreement():
    # One source cell receives two image pixels with equal weights.
    indices = torch.tensor([[0, 0], [0, 1]])
    values = torch.tensor([1.0, 1.0])
    mapping = torch.sparse_coo_tensor(indices, values, (1, 4)).coalesce()
    module = SourcePlaneIntensityConsistency(
        mapping,
        source_shape=(1, 1),
        min_effective_contributors=1.5,
        variance_weight=1.0,
        normalize_by_source_power=False,
    )

    source = torch.tensor([[[[0.5]]]])
    consistent_image = torch.tensor([[[[0.5, 0.5], [0.0, 0.0]]]])
    inconsistent_image = torch.tensor([[[[0.0, 1.0], [0.0, 0.0]]]])

    consistent_loss, consistent_diag = module(consistent_image, source)
    inconsistent_loss, inconsistent_diag = module(inconsistent_image, source)
    assert consistent_loss.item() < 1e-8
    assert inconsistent_loss.item() > 0.0
    assert inconsistent_diag["variance_loss"] > consistent_diag["variance_loss"]

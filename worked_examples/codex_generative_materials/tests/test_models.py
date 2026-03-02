import torch

from microgen.models.ddpm import DiffusionSchedule, TinyUNet, q_sample
from microgen.models.gan import DCDiscriminator, DCGenerator
from microgen.models.vae import ConvVAE


def test_vae_forward_shape() -> None:
    model = ConvVAE(latent_dim=16)
    x = torch.randn(4, 1, 64, 64)
    recon, mu, logvar = model(x)
    assert recon.shape == x.shape
    assert mu.shape == (4, 16)
    assert logvar.shape == (4, 16)


def test_gan_shapes() -> None:
    gen = DCGenerator(latent_dim=32)
    disc = DCDiscriminator()
    z = torch.randn(4, 32)
    fake = gen(z)
    logits = disc(fake)
    assert fake.shape == (4, 1, 64, 64)
    assert logits.shape == (4,)


def test_ddpm_unet_and_q_sample_shapes() -> None:
    model = TinyUNet(base_channels=16)
    schedule = DiffusionSchedule(timesteps=16)
    x0 = torch.randn(4, 1, 64, 64)
    t = torch.randint(0, 16, (4,))
    xt, noise = q_sample(x0, t, schedule)
    pred = model(xt, t)
    assert xt.shape == x0.shape
    assert noise.shape == x0.shape
    assert pred.shape == x0.shape

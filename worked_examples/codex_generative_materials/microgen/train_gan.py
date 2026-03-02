"""Train a minimal DCGAN on synthetic microstructures."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import optim
from torch.nn import functional as F

from microgen.models.gan import DCDiscriminator, DCGenerator
from microgen.train_common import ensure_output_dirs, load_training_tensor, make_dataloader, pick_device, save_grid, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GAN on synthetic microstructures.")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--kind", choices=["porous", "precipitate", "mixed"], default="mixed")
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--g-lr", type=float, default=2e-4)
    parser.add_argument("--d-lr", type=float, default=1.5e-4)
    parser.add_argument("--d-steps", type=int, default=1)
    parser.add_argument("--instance-noise-std", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fast", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = pick_device()
    ckpt_dir, out_dir = ensure_output_dirs()

    epochs = args.epochs if args.epochs is not None else (10 if args.fast else 25)
    batch_size = args.batch_size if args.batch_size is not None else (64 if args.fast else 128)
    num_samples = min(args.num_samples, 384) if args.fast else args.num_samples

    x, metadata = load_training_tensor(
        data_path=args.data_path,
        num_samples=num_samples,
        image_size=args.image_size,
        kind=args.kind,
        seed=args.seed,
        normalize_to_neg1=True,
    )
    loader = make_dataloader(x, batch_size=batch_size, shuffle=True)

    gen = DCGenerator(latent_dim=args.latent_dim).to(device)
    disc = DCDiscriminator().to(device)
    opt_g = optim.Adam(gen.parameters(), lr=args.g_lr, betas=(0.5, 0.999))
    opt_d = optim.Adam(disc.parameters(), lr=args.d_lr, betas=(0.5, 0.999))

    print(f"device={device} samples={len(x)} epochs={epochs} batch_size={batch_size}")
    for epoch in range(1, epochs + 1):
        gen.train()
        disc.train()
        g_loss_epoch = 0.0
        d_loss_epoch = 0.0
        batches = 0
        noise_std = args.instance_noise_std * (1.0 - (epoch - 1) / max(1, epochs - 1))
        for (real,) in loader:
            real = real.to(device)
            bsz = real.size(0)
            last_d_loss = torch.tensor(0.0, device=device)

            for _ in range(args.d_steps):
                opt_d.zero_grad(set_to_none=True)
                real_for_disc = real
                if noise_std > 0:
                    real_for_disc = torch.clamp(real + noise_std * torch.randn_like(real), -1.0, 1.0)
                real_scores = disc(real_for_disc)

                z = torch.randn(bsz, args.latent_dim, device=device)
                fake_imgs = gen(z)
                fake_for_disc = fake_imgs.detach()
                if noise_std > 0:
                    fake_for_disc = torch.clamp(fake_for_disc + noise_std * torch.randn_like(fake_for_disc), -1.0, 1.0)
                fake_scores = disc(fake_for_disc)

                # Hinge discriminator loss.
                d_loss = F.relu(1.0 - real_scores).mean() + F.relu(1.0 + fake_scores).mean()
                d_loss.backward()
                opt_d.step()
                last_d_loss = d_loss.detach()

            opt_g.zero_grad(set_to_none=True)
            z = torch.randn(bsz, args.latent_dim, device=device)
            fake_imgs = gen(z)
            # Hinge generator loss.
            g_loss = -disc(fake_imgs).mean()
            g_loss.backward()
            opt_g.step()

            g_loss_epoch += float(g_loss.item())
            d_loss_epoch += float(last_d_loss.item())
            batches += 1

        print(f"[epoch {epoch:03d}] d_loss={d_loss_epoch / batches:.6f} g_loss={g_loss_epoch / batches:.6f}")

    gen.eval()
    with torch.no_grad():
        z = torch.randn(16, args.latent_dim, device=device)
        samples = gen(z).cpu()
        save_grid(samples, out_dir / "gan_samples.png", normalize=True, value_range=(-1, 1), nrow=4)

    ckpt_path = ckpt_dir / "gan.pt"
    torch.save(
        {
            "generator_state": gen.state_dict(),
            "discriminator_state": disc.state_dict(),
            "config": {
                "latent_dim": args.latent_dim,
                "epochs": epochs,
                "batch_size": batch_size,
                "num_samples": int(len(x)),
                "metadata": metadata,
            },
        },
        ckpt_path,
    )
    print(f"saved_checkpoint={Path(ckpt_path)}")
    print(f"saved_outputs={out_dir / 'gan_samples.png'}")


if __name__ == "__main__":
    main()

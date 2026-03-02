"""Train a minimal convolutional VAE on synthetic microstructures."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import optim

from microgen.models.vae import ConvVAE, vae_loss
from microgen.train_common import ensure_output_dirs, load_training_tensor, make_dataloader, pick_device, save_grid, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VAE on synthetic microstructures.")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--kind", choices=["porous", "precipitate", "mixed"], default="mixed")
    parser.add_argument("--num-samples", type=int, default=768)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--kl-weight", type=float, default=5e-4)
    parser.add_argument("--kl-warmup-epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fast", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = pick_device()
    ckpt_dir, out_dir = ensure_output_dirs()

    epochs = args.epochs if args.epochs is not None else (10 if args.fast else 30)
    batch_size = args.batch_size if args.batch_size is not None else (64 if args.fast else 128)
    num_samples = min(args.num_samples, 384) if args.fast else args.num_samples

    x, metadata = load_training_tensor(
        data_path=args.data_path,
        num_samples=num_samples,
        image_size=args.image_size,
        kind=args.kind,
        seed=args.seed,
        normalize_to_neg1=False,
    )
    loader = make_dataloader(x, batch_size=batch_size, shuffle=True)

    model = ConvVAE(latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"device={device} samples={len(x)} epochs={epochs} batch_size={batch_size}")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        kl_weight = args.kl_weight * min(1.0, epoch / max(1, args.kl_warmup_epochs))
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon, mu, logvar = model(batch_x)
            loss, recon_loss, kl = vae_loss(recon, batch_x, mu, logvar, kl_weight=kl_weight)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * batch_x.size(0)
        avg_loss = total_loss / len(x)
        recon_scalar = float(recon_loss.detach().cpu())
        kl_scalar = float(kl.detach().cpu())
        print(
            f"[epoch {epoch:03d}] loss={avg_loss:.6f} recon={recon_scalar:.6f} "
            f"kl={kl_scalar:.6f} kl_w={kl_weight:.6f}"
        )

    model.eval()
    with torch.no_grad():
        preview = x[:8].to(device)
        # Use deterministic reconstruction (decode(mu)) for visualization stability.
        mu, _ = model.encode(preview)
        recon = model.decode(mu).cpu()
        combined = torch.cat([preview.cpu(), recon], dim=0)
        save_grid(combined, out_dir / "vae_recon_grid.png", normalize=False, nrow=8)

        z = torch.randn(16, args.latent_dim, device=device)
        samples = model.decode(z).cpu()
        save_grid(samples, out_dir / "vae_samples.png", normalize=False, nrow=4)

    ckpt_path = ckpt_dir / "vae.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
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
    print(f"saved_outputs={out_dir / 'vae_recon_grid.png'}, {out_dir / 'vae_samples.png'}")


if __name__ == "__main__":
    main()

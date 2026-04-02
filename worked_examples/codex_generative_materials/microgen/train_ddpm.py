"""Train a minimal DDPM on synthetic microstructures."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn, optim

from microgen.models.ddpm import DiffusionSchedule, TinyUNet, p_sample_loop, q_sample
from microgen.train_common import ensure_output_dirs, load_training_tensor, make_dataloader, pick_device, save_grid, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DDPM on synthetic microstructures.")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--kind", choices=["porous", "precipitate", "mixed"], default="mixed")
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fast", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = pick_device()
    ckpt_dir, out_dir = ensure_output_dirs()

    epochs = args.epochs if args.epochs is not None else (8 if args.fast else 20)
    batch_size = args.batch_size if args.batch_size is not None else (64 if args.fast else 128)
    timesteps = args.timesteps if args.timesteps is not None else (100 if args.fast else 300)
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

    model = TinyUNet(base_channels=32).to(device)
    schedule = DiffusionSchedule(timesteps=timesteps).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    print(
        f"device={device} samples={len(x)} epochs={epochs} batch_size={batch_size} timesteps={timesteps}"
    )
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        batches = 0
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            t = torch.randint(0, timesteps, (batch_x.size(0),), device=device)
            x_t, noise = q_sample(batch_x, t, schedule)
            pred_noise = model(x_t, t)
            loss = criterion(pred_noise, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running += float(loss.item())
            batches += 1
        print(f"[epoch {epoch:03d}] noise_mse={running / batches:.6f}")

    model.eval()
    with torch.no_grad():
        samples = p_sample_loop(model, schedule, shape=(16, 1, args.image_size, args.image_size), device=device).cpu()
        save_grid(samples, out_dir / "ddpm_samples.png", normalize=True, value_range=(-1, 1), nrow=4)

    ckpt_path = ckpt_dir / "ddpm.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {
                "epochs": epochs,
                "batch_size": batch_size,
                "num_samples": int(len(x)),
                "timesteps": timesteps,
                "metadata": metadata,
            },
        },
        ckpt_path,
    )
    print(f"saved_checkpoint={Path(ckpt_path)}")
    print(f"saved_outputs={out_dir / 'ddpm_samples.png'}")


if __name__ == "__main__":
    main()

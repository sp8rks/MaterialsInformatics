import os
import argparse
import numpy as np
import torch
from imagen-pytorch import Unet, SRUnet256, Imagen, ImagenTrainer


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, help="Path to where the dataset is stored.")
parser.add_argument("--save_path", type=str, help="Path to where the models is to be stored.")
parser.add_argument("--model_path", type=str, help="Path to where the models is stored.")
parser.add_argument("--unet_number", type=str, help="Number of the Unet that is to be trained.")

def load_real_samples(data_path: str) -> np.ndarray:
    """ Loads in the tensor of real samples.

      Args:
        data_path: Path to the dataset.
      Returns:
        Tensor of real samples. Shape = (num_samples, 64, 64, 4).
    """
    data_tensor = np.load(data_path)
    return data_tensor

def train_imagen(path, dataset, train_unet_number, model_path: None):
    unet1 = Unet(
    dim = 256,
    dim_mults = (1, 2, 4),
    num_resnet_blocks = 3,
    layer_attns = (False, True, True),
    layer_cross_attns = (False, True, True),
    use_linear_attn = True,
    channels = 4,
    channels_out = 4
    )


    unet2 = SRUnet256(
    dim = 256,
    dim_mults = (1, 2, 4),
    num_resnet_blocks = (2, 4, 8),
    layer_attns = (False, False, True),
    layer_cross_attns = (False, False, True),
    channels = 4,
    channels_out = 4
    )

    imagen = Imagen(
    condition_on_text = False,
    unets = (unet1, unet2),
    image_sizes = (32, 64),
    timesteps = 1000,
    channels = 4,
    )

    trainer = ImagenTrainer(imagen).cuda()

    if model_path is None:
        train_unet_number = 1
    else:
        trainer.load(model_path)
    low = 10000
    loss = []
    for i in range(250000):
        if len(loss) > 0 and loss[-1] < low:
            low = loss[-1]
            trainer.save(path)
        ix = np.random.randint(0, dataset.shape[0], 4)
        training_images = dataset[ix]
        training_images = torch.from_numpy(training_images).float().cuda()
        loss.append(trainer(training_images, unet_number = train_unet_number))
        trainer.update(unet_number = train_unet_number)

def main():
    args = parser.parse_args()
    dataset = load_real_samples(args.data_path)
    train_imagen(args.path, dataset, args.unet_number, args.model_path)

if __name__ == "__main__":
    main()
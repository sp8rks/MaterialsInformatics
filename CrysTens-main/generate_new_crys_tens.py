import numpy as np
import argparse
from imagen-pytorch import Unet, SRUnet256, Imagen, ImagenTrainer
import torch
import tensorflow as tf
import random
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, help="Path to where the trained model is.")
parser.add_argument("--num_crys_tens", type=int, help="The number of crystals to generate.")
parser.add_argument("--crys_tens_path", type=str, help="Path to where the stacked CrysTens should be stored.")

def generate_latent_points(latent_dim: int, n_samples:int) -> np.ndarray:
    """Generates a random array to be used by the generator.

    Args:
      latent_dim: The dimension of the latent space.
      n_samples: The number of samples requested.

    Returns:
      An array of random latent variables. Shape = (n_samples, latent_dim).
    """
    x_input = random.randn(latent_dim*n_samples)
    x_input = x_input.reshape(n_samples,latent_dim)
    return x_input

def generate_with_diffusion(model_path, num_crys_tens, crys_tens_path):
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

    trainer.load(model_path)

    samp = trainer.sample(batch_size = num_crys_tens)
    np_samp = samp.cpu().detach().numpy()
    np.save(crys_tens_path, np_samp)

def generate_with_gan(model_path, num_crys_tens, crys_tens_path):
    model = tf.keras.models.load_model(model_path)
    samp = np.zeros((num_crys_tens, 64, 64, 4))
    for i in range(num_crys_tens):
        rand = generate_latent_points(128, 1)
        crys = model.predict(rand)
        samp[i, :, :, :] = np.copy(crys[0, :, :, :, 0])
    np.save(crys_tens_path, samp)

def main():
    args = parser.parse_args()
    if ".pt" in args.model_path:
        generate_with_diffusion(args.model_path, args.num_crys_tens, args.crys_tens_path) 
    else:
        generate_with_gan(args.model_path, args.num_crys_tens, args.crys_tens_path) 

if __name__ == "__main__":
    main()
import os
import argparse
import numpy as np
from numpy import random
from typing import Tuple
import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv3D, Conv3DTranspose, Dropout, Input
from tensorflow.keras.layers import Flatten, LeakyReLU, ReLU, BatchNormalization, Reshape

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, help="Path to where the dataset is stored.")
parser.add_argument("--save_path", type=str, help="Path to where the models and loss history is stored.")

def conv_norm(x: keras.engine.keras_tensor.KerasTensor, units: int, 
              filter: Tuple[int, int, int], stride : Tuple[int, int, int], 
              discriminator: bool = True
              ) -> keras.engine.keras_tensor.KerasTensor:
  """Applies either a convolution or transposed convolution, normalization, and 
     an activation function.

  Args:
    x: The previous keras tensor passed into the 
    convolutional/normalization layer
    units: The number of channels in the convolutional layer
    filter: The filter size of the convolutional layer
    stride: The stride size of the convolutional layer
    discriminator: Whether conv_norm is present in the discriminator. If true,
    a convolution (as opposed to a transposed convolution) will be applied and
    the activation function will be LeakyReLU instead of ReLU. 

  Returns:
    The keras tensor after the convolution, normalization, and activation 
    function.
  """
  if discriminator:
    activation_function = LeakyReLU(alpha = 0.2)
    conv = Conv3D(units, filter, strides = stride, padding = 'valid')
  else:
    activation_function = ReLU()
    conv = Conv3DTranspose(units, filter, strides = stride, padding = 'valid')
  x = conv(x)
  x = BatchNormalization()(x)
  x = activation_function(x) 
  return x

def dense_norm(x: keras.engine.keras_tensor.KerasTensor, units: int, 
               discriminator: bool) -> keras.engine.keras_tensor.KerasTensor:
  """Applies a dense layer, normalization, and an activation function.

  Args:
    x: The previous keras tensor passed into the dense/normalization layer
    units: The number of units in the dense layer
    discriminator: Whether dense_norm is present in the discriminator. If true,
    the activation function will be LeakyReLU instead of ReLU. 

  Returns:
    The keras tensor after the dense layer, normalization, and activation 
    function.
  """
  if discriminator:
    activation_function = LeakyReLU(alpha = 0.2)
  else:
    activation_function = ReLU()
  x = Dense(units)(x)
  x = BatchNormalization()(x)
  x = activation_function(x)
  return x

def define_discriminator(in_shape: Tuple[int, int, int, int] = (64, 64, 4, 1)
) -> keras.engine.functional.Functional:
    """Constructs the discriminator in the GAN. Uses a combination of dense
    and convolutional layers.

    Args:
      in_shape: The shape of the input keras tensor. 

    Returns:
      The discriminator model.
    """
 
    tens_in = Input(shape=in_shape, name="input")
 
    y = Flatten()(tens_in)
    y = dense_norm(y, 1024, True) 
    y = dense_norm(y, 1024, True)
    y = dense_norm(y, 1024, True)
    y = dense_norm(y, 1024, True)

    x = conv_norm(tens_in, 32, (1,1,1), (1,1,1), True)
    x = conv_norm(x, 32, (3,3,1), (1,1,1), True)  
    x = conv_norm(x, 32, (3,3,1), (1,1,1), True)  
    x = conv_norm(x, 32, (3,3,1), (1,1,1), True)  
    x = conv_norm(x, 64, (3,3,1), (1,1,1), True) 
    x = conv_norm(x, 64, (7,7,1), (1,1,1), True)
    x = conv_norm(x, 64, (7,7,1), (1,1,1), True)
    x = conv_norm(x, 64, (7,7,1), (1,1,1), True)
    x = conv_norm(x, 64, (7,7,1), (1,1,1), True)

    z = Reshape((32, 32, 1, 1))(y)
    x = z + x

    y = dense_norm(y, 9, True) 
 
    x = conv_norm(x, 128, (5,5,2), (5,5,1), True)
    x = conv_norm(x, 256, (2,2,2), (2,2,2), True)  
 
    z = Reshape((3, 3, 1, 1))(y)
    x = z + x
 
    x = Flatten()(x)
    x = Dropout(0.25)(x)
 
    disc_out = Dense(1, activation = "sigmoid")(x)
    model = Model(inputs=tens_in, outputs=disc_out)
    opt = Adam(learning_rate = 1e-5)
    model.compile(loss = 'binary_crossentropy', optimizer = opt,metrics = ['accuracy'])

    return model
 
 
def define_generator(latent_dim: int) -> keras.engine.functional.Functional:
    """Constructs the generator in the GAN. Uses a combination of dense
    and transposed convolutional layers.

    Args:
      latend_dim: The dimension of latent space. 

    Returns:
      The generator model.
    """
    n_nodes = 16 * 16 * 4
 
    noise_in = Input(shape=(latent_dim, ), name="noise_input")

    y = dense_norm(noise_in, 484, False)
    y = dense_norm(y, 484, False)
    
    x = dense_norm(noise_in, n_nodes, False)
    x = Reshape((16,16, 4, 1))(x)
    x = conv_norm(x, 256, (3,3,3), (1,1,1), False)
    x = conv_norm(x, 128, (3,3,3), (1,1,1), False)
    x = conv_norm(x, 128, (3,3,3), (1,1,1), False)
 
    z = Reshape((22, 22, 1, 1))(y)
    x = z + x

    y = dense_norm(y, 784, False)
    y = dense_norm(y, 784, False)
    y = dense_norm(y, 784, False)
 
    x = conv_norm(x, 128, (3,3,3), (1,1,1), False)
    x = conv_norm(x, 64, (3,3,3), (1,1,1), False) 
    x = conv_norm(x, 64, (3,3,3), (1,1,1), False) 
 
    z = Reshape((28, 28, 1, 1))(y)
    x = z + x

    y = dense_norm(y, 1024, False)
    y = dense_norm(y, 1024, False)
 
    x = conv_norm(x, 64, (3,3,3), (1,1,1), False) 
    x = conv_norm(x, 64, (3,3,3), (1,1,1), False) 
 
    z = Reshape((32, 32, 1, 1))(y)
    x = z + x

    y = dense_norm(y, 4096, False)
 
    x = conv_norm(x, 32, (2,2,2), (2,2,2), False)   
 
    z = Reshape((64, 64, 1, 1))(y)
    x = z + x
 
    outMat = Conv3D(1,(1,1,10), activation = 'sigmoid', strides = (1,1,10), padding = 'valid')(x)
 
    model = Model(inputs=noise_in, outputs=outMat)
    return model

def define_gan(generator: keras.engine.functional.Functional, 
               discriminator: keras.engine.functional.Functional
               ) -> keras.engine.functional.Functional:
    
    """Constructs the GAN. Freezes the discriminator weights to train generator.

    Args:
      generator: The generator model.
      discriminator: The discriminator model.
    
    Returns:
      The GAN model.
    """
    discriminator.trainable = False
    model = Sequential()
    
    model.add(generator)
    model.add(discriminator)
    
    opt = Adam(learning_rate = 1e-5)
    model.compile(loss = 'binary_crossentropy', optimizer = opt)
    return model

def load_real_samples(data_path: str) -> np.ndarray:
    """ Loads in the tensor of real samples.

      Args:
        data_path: Path to the dataset.
      Returns:
        Tensor of real samples. Shape = (num_samples, 64, 64, 4).
    """
    data_tensor = np.load(data_path)
    return np.reshape(data_tensor, (data_tensor.shape[0], 64, 64, 4))

def generate_real_samples(dataset: np.ndarray, n_samples: int
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly selects a number of samples from the real dataset.

    Args:
      dataset: The dataset of real samples.
      n_samples: The number of samples requested.

    Returns:
      The randomly selected samples from the dataset and a vector of ones
      to indicate that the samples are "real."
    """
    ix = random.randint(0,dataset.shape[0],n_samples)
    X = dataset[ix]
    y = np.ones((n_samples,1))
    return X,y
    
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
 
def generate_fake_samples(generator: keras.engine.functional.Functional, 
                          latent_dim: int, n_samples: int
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """Generates fake samples from the generator.

    Args:
      generator: The generator model.
      latent_dim: The dimension of latent space.
      n_samples: The number of samples requested.

    Returns:
      The generated samples and a vector of zeros to indicate that the samples
      are "fake."
    """
    x_input = generate_latent_points(latent_dim,n_samples)
    X = generator.predict(x_input)
    y = np.zeros((n_samples,1))
    return X,y

def train(g_model: keras.engine.functional.Functional,
          d_model: keras.engine.functional.Functional,
          gan_model: keras.engine.functional.Functional,
          dataset: np.ndarray, latent_dim: int, save_path: str,
          n_epochs: int = 100, n_batch: int = 64) -> None:
    """Trains the GAN model.

    Args:
      g_model: The generator model.
      d_model: The discriminator model.
      gan_model: The GAN model.
      dataset: The dataset of real samples.
      latent_dim: The dimension of the latent space.
      n_epochs: The number of epochs to train for.
      n_batch: The batch size.
      save_path
    """
    bat_per_epoch = int(dataset.shape[0]/n_batch)
    d_loss_real_list = []
    d_loss_fake_list = []
    g_loss_list = []
    for i in range(n_epochs):
        for j in range(bat_per_epoch//2):
            X_real,y_real = generate_real_samples(dataset, n_batch)
            d_loss_real,_ = d_model.train_on_batch(X_real, y_real)
            X_fake,y_fake = generate_fake_samples(g_model, latent_dim, n_batch)
            d_loss_fake,_ = d_model.train_on_batch(X_fake, y_fake)
            X_gan = generate_latent_points(latent_dim, n_batch)
            y_gan = np.ones((n_batch,1))
            g_loss = gan_model.train_on_batch(X_gan,y_gan)
        
        d_loss_real_list.append(d_loss_real)
        d_loss_fake_list.append(d_loss_fake)
        g_loss_list.append(g_loss)

        g_model.save(os.path.join(save_path, 'generator'))
        d_model.save(os.path.join(save_path, 'discriminator'))
        np.savetxt(os.path.join(save_path, 'd_loss_real_list'),d_loss_real_list)
        np.savetxt(os.path.join(save_path, 'd_loss_fake_list'),d_loss_fake_list)
        np.savetxt(os.path.join(save_path, 'g_loss_list'),g_loss_list)

def main():
    args = parser.parse_args()
    latent_dim = 128
    discriminator = define_discriminator()
    generator = define_generator(latent_dim)
    gan_model = define_gan(generator,discriminator)
    dataset = load_real_samples(args.data_path)
    train(generator, discriminator, gan_model,dataset, latent_dim, args.save_path)

if __name__ == "__main__":
    main()

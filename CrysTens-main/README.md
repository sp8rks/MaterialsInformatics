# CrysTens
CrysTens is a representation for storing crystal structure information that is originally in the form of Crystallographic Information Files (CIFs). CrysTens is a tensor of size 64x64x4 that can be used in any type of machine learning application involving crystal structures. This repository houses code for creating a stack of CrysTens', using the stack to train either a Vanilla Generative Adversarial Network (GAN), a Wasserstein GAN, or a diffusion model from https://github.com/lucidrains/imagen-pytorch/tree/main/imagen_pytorch. Once a model has been trained, newly generated CIFs can be created from a stack of generated CrysTens'. The details of the CrysTens representation, model training, and model analysis in the field of material discovery can be found here: https://chemrxiv.org/engage/chemrxiv/article-details/63694d64fbfd387c25d2d395.

## Making a Stacked CrysTens
In order to train a CrysTens generative model for material discovery, a stack of CrysTens' is required. Using ```get_stacked_crys_tens.py``` and a Crystal Dictionary, any size of Stacked CrysTens can be received. 

```
python get_stacked_crys_tens.py --crys_dict_path=Data/CrystalDictionary.jsonl --num_examples=500 --crys_tens_path=Data/StackedCrysTensor.npy
```

### Crystal Dictionary
A crystal dictionary is any ```.jsonl``` file that lists crystal structures in the following form.

```
{"Crystal Structure Name": {"a": ..., "b": ..., "c": ..., "alpha": ..., "beta": ..., "gamma": ..., "sg": ..., "siteList": [[Atomic Number 1, [X_1, Y_1, Z_1]], [Atomic Number 2, [X_2, Y_2, Z_2]], ...}}
```

Here is an example.
```
{"Ca4MgPd-1934448.cif": {"a": 14.545, "b": 14.545, "c": 14.545, "alpha": 90.0, "beta": 90.0, "gamma": 90.0, "sg": 216, "siteList": [[20, [0.56266, 0.25, 0.25]], [20, [0.43734, 0.75, 0.25]], ...}
```

## Training a generative model
Once a Stacked CrysTens has been created, we can train a generative model to produce synthetic CrysTens'. The options are a Vanilla GAN (lowest performance), a Wasserstein GAN (intermediate performance), and a diffusion model (highest) performance. 

### Vanilla GAN
```
python train_vanilla_gan.py --data_path=Data/StackedCrysTensor.npy --save_path=Data/
```

### Wasserstein GAN
```
python train_wasserstein_gan.py --data_path=Data/StackedCrysTensor.npy --save_path=Data/
```

### Diffusion Model
```
python train_diffusion_model.py --data_path=Data/StackedCrysTensor.npy --save_path=Data/diffusion.pt --unet_number=1
```
followed by
```
python train_diffusion_model.py --data_path=Data/StackedCrysTensor.npy --save_path=Data/diffusion.pt --model_path=Data/diffusion.pt --unet_number=2
```

## Generating CrysTens'
Once a trained generative model is obtained, we can generate a stack of synthetic CrysTens' using ```generate_new_crys_tens.py```.

```
python generate_new_crys_tens.py --model_path=Data/diffusion.pt --num_crys_tens=200 --crys_tens_path=Data/GenStackedCrysTens.npy
```

## Getting CIFs and statistics
With a stack of generated CrysTens', generate_crystal_statistics.py can be used to turn the CrysTens' into CIFs as well as collect statistics about the performance of the generative model that produced them. 

```
python generate_crystal_statistics --crys_tens_path=Data/GenStackedCrysTens.npy --cif_folder=Data/ --stats_folder=Data/
```
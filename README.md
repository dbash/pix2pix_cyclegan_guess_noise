

# Defending CycleGAN from Self-Adversarial Attack

In this repository you can find the extension of the original 
implementation of CycleGAN in pytorch. This extension contains 
two defense techniques against the self-adversarial attacks 
performed by the unsupervised image-to-image translation methods 
that hold the cycle-consistency property. 
More information on self-adversarial attacks can be found in 
our paper. 

You can find the instructions on how to train and test the model 
with additional guess loss or with Gaussian noise 
[here](docs/howto_guess.md) and [here](docs/howto_noisy.md) 
respectively.

In order to reproduce our results, you can find all the configuration
 files in [configs](configs) directory.
 
<img src='http://cs-people.bu.edu/dbash/honest_gan/result_figs/gta2segm_guess.png' align="right" width=800>

<br><br><br>


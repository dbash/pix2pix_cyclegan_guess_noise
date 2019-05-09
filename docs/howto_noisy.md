#Noisy CycleGAN

The idea behind noisy CycleGAN is to train the model to generate reconstruction
results robust to high-frequency perturbations which lie in the same frequency spectre with 
the self-adversarial structured noise. 
After the input image has been mapped to the target domain, the 
Gaussian noise is added to the translation result before reconstruction 
of the input. The noisy CycleGAN model is stored in the file 
[cycle_gan_noisy_model.py](../models/cycle_gan_noisy_model.py).

In addition to the parameters of the [original CycleGAN](tips.md), 
noisy CycleGAN an additional parameter *noise_std* that sets the standard 
deviation of a zero-mean Gaussian nose added to the translated image 
before reconstruciton.

The command for training the model is: 
```bash
python train.py --model cycle_gan_noisy --dataroot /path/to/data --noise_std *some_double less than 1*
```
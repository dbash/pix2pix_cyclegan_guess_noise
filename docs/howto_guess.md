#Guess Discriminator 

Guess discriminator is an additional network that receives as input 
a concatenation of the input image and its reconstruction in a random order. 
The goal of the guess discriminator is to correctly predict whether 
the first image in the concatenation is fake or not. 
Guess discriminator aids CycleGAN in preventing the self-adversarial
attack by mimicking the adversarial training defense approach widely used 
to make DNNs more robust to adversarial structured noise.

##Training and testing
The training and testing procedures of the CycleGAN with Guess Discriminator are 
very similar to those of the original CycleGAN (basic instructions and recommendations on how to train the original
CycleGAN that could be found [here](tips.md)).
However there are a few details.
1. Guess-based model is stored in the file [cycle_gan_guess_model.py](../models/cycle_gan_guess_model.py).
In order to train this model, call:
```bash
python train.py --model cycle_gan_guess --dataroot /path/to/data --lambda_guess *some_double*
```
2. Parameter *lambda_guess* defines a weight of the guess discriminator loss.
 In our experiments, we set it close to or same as 
*lambda_A* and *lambda_B*.
3. The results of our experiments showed clear improvement on the model's
performance when the weight on the cycle-consistency loss reflects the 
relationship between the goal and target domains. For example, if domain A 
is *many* (like real frames in the GTA V experiments) and domain B is 
*one* (e.g. semantic segmentation), then lambda_A should be less than 
lambda_B. 
# AudioVision
AudioVision Repository 
Team : Sight&Sound

Relevant Directory :  `final/` and `sahil/`

Code for Sound to Sound Architecture : `final/models.py` (VAE) and `final/simplemodel.py` (Autoencoder) 
Files which load data and run these experiments : `final/vaeExp.py` and `final/autoEncoderExp.py` (Autoencoder)
These models take sound in the form of a spectogram (48x128 - downsampled from 192x512) and try to regenerate the same spectogram after forcing the input down to a latent space (128 dimensional)

Code for Image to Sound Architecture : `imsimple.py`
Files which load data and run this experiment : `final/im2sound.py` and `final/im2soundPretrain.py`
The first file uses a pretrained resnet to extract image features and then applies loss to do the following:
* Minimise MSE loss between latent space representation of paired image and sound
* Minimise MSE between ground truth and reconstructed image using input sound spectogram
* Minimise MSE between ground truth and reconstructed image using image features
The second file uses a pretrained sound encoder and decoder (trained using `autoEncoderExp.py`) to start with.

Code to run : `python exp2.py` (to use custom data, use `data` variable line 21 of exp2.py (`sahil/exp2.py`)) 


## Requirements
```
pytorch
progress
librosa
numpy, pickle and some standard libraries
```

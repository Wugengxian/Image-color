# Image Colorization of pix2pix with referenced graph

### 1. train.py and test.py
Use this to train the modified network.

```
python train.py --training_dir <the datasets> --preference
```

--preference means whether you need to train with reference image 

Use this this to train the pix2pix without reference image

```
python train.py --training_dir <the datasets> --gan_g
```

### 2. Model directory

this directory is for you to obtain the pix2pix network framework.

### 3. Process_for_GAN directory

this directory is for you to train the model of origin net based on essay *Colorful Image Colorization*, skip connection model, GAN model and cycle GAN model. For more information, step into this directory for more details.

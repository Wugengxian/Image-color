# Image Colorization via eccv16

### 1. Obtain Dataset

```
python preprocess_img.py -d <train_directory>
```

This command will get the preprocessed dataset and save the array to pics.npy and labels.npy automatically. For the training process, the model will auto load these two files

### 2. train the model

```
python <model>.py
```

The model can be chosen with original_net.py, skip_connection.py, gan.py, cycleGAN.py. The model will train automatically. Then the trained model can be saved into the pkl files.

### 3. Test result

```
python test.py -i <test_img> -m <model> -s <saved_img>
```

You can set test image and saved image paths. For the model, you will enter the name of the pkl file.
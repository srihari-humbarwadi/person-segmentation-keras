# Unet
A segmentation network based on the original unet architecture with the following changes
 - Use of trainable Conv2DTranspose layers instead of UpSampling2D
 - layer inputs batch normalized
### Main blocks
```
def conv_block(tensor, nfilters, size=3, padding='same', initializer="he_normal"):
    x = Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(tensor)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def deconv_block(tensor, residual, nfilters, size=3, padding='same', strides=(2, 2)):
    y = Conv2DTranspose(nfilters, kernel_size=(size, size), strides=strides, padding=padding)(tensor)
    y = concatenate([y, residual], axis=3)
    y = conv_block(y, nfilters)
    return y
```

## Results
![alt text](outputs/cropped/output_27.png)
![alt text](outputs/cropped/output_39.png)
![alt text](outputs/cropped/output_49.png)
![alt text](outputs/cropped/output_50.png)
![alt text](outputs/cropped/output_58.png)

## Todo
- add dropout
- add option for residual connections

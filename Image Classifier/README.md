### CNN HW

## HANDWRITTEN DIGITS RECOGNITION

#### try to make ROC model

### The models
####The tutorial model 
Layers : 

        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10)

###### look at which digits are wrong and which digits are right.

* Convolution Layer w/ RELU activation
    * Convolution is the first layer  to extract features from an input image.
    Convolution preserves the relationship between pixels by learning image features using small
    squares of input data. This model only has one feature detection layer. Our layer uses a 3*3 kernel.
* Flattening Layer
    * 
* Dense Layer 1 w/ RELU activation
* Dense Layer 2 w/ Linear activation



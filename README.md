### TensorflowJS GAN
A simple Generative Adversarial Network example generating hand-written digits.


#### Architecture & algorithm
Both generator and discriminator use densely connected layers, with leaky-Relu for the hidden layer activations. Output uses tanh and sigmoid, respectively.

Each training batch run looks something like this.
```
G := generator()
DReal := discriminator()
DFake := discriminator(G)

DReal_loss := 1 - DReal(next_image_batch)
DFake_loss := 0 - DFake(random_seed_batch)
DLoss := DReal_loss + DFake_loss
...backprop on discriminator variables ...

GLoss := 1 - DFake(random_seed_batch)
...backprop on generator variables ...
```


#### Source files
- gan.js: GAN code and hyper params
- image-util.js: Image utility
- data.js: Used to load MNIST dataset into compatible tensors, taken from tensorflow examples.
- model.js: Pre-trained weights/biases for the _Generator_ network.

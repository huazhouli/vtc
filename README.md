This is a application of vgg convolutional neural network using keras api, which network training and testing are both included.    

The finetuning changed the input size of the network from [224 224] to [32 32] because of the cifar-10 dataset picture size we use. And the output of softmax layer is changed from 2 to ten beacase of the dataset differences, too.  

More info can be found from https://github.com/1297rohit/VGG16-In-Keras

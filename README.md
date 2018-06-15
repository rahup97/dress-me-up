# dress-me-up
Implementation of CNN architectures on the Fashion MNIST dataset using Keras. Experimented with different types of architectures and various combination of hyperparameters - such as variation in the dropout values, number of filters, size of the filters, pooling layer size and stride and learning rate.

The results of these experiments along with various architectures can be found in the previous commits made, and can be found in the previous commits.

## Fashion MNIST
The MNIST dataset has been one of the most overused datasets in Deep Learning. The normalized dataset of handwritten digits comprised of 60,000 training examples, accompanied by 10,000 test samples. These grayscale images are 784 pixels each, of 28x28 dimension to be specific. They are individual handwritten digits, each associated with a label ranging between 0 to 9. It has been used too extensively, and has also been quoted as being too 'easy' and is not an accurate reflection of actual Computer Vision problems.

Hence, the team at Zalando research have come up with a new dataset called Fashion MNIST, that is comprised of 55,000 training samples and 10,000 test samples, very much similar to the handwritten dataset. Here also, each image is in grayscale and has dimensions of 28x28 pixels. The images essentially belong to 10 different classes such as shirts, trousers, dresses, sneakers, etc.

You can find the dataset directly in kaggle [Fashion MNiST](https://www.kaggle.com/zalando-research/fashionmnist/)


## Run the code
After downloading the dataset, give the required path to the training and test csv files and run the following code
`python model.py`

# Download the MNIST datasets from http://yann.lecun.com/exdb/mnist/
# and extract them
echo "Downloading training images..."
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz 
echo "Downloading training labels..."
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
echo "Downloading testing images..."
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
echo "Downloading testing labels..."
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

mv train-images-idx3-ubyte.gz train-images.gz
mv train-labels-idx1-ubyte.gz train-labels.gz
mv t10k-images-idx3-ubyte.gz test-images.gz
mv t10k-labels-idx1-ubyte.gz test-labels.gz

gunzip train-images.gz
gunzip train-labels
gunzip test-images
gunzip test-labels

exit 0

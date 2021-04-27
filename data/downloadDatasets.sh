# Download the MNIST datasets from http://yann.lecun.com/exdb/mnist/
# and extract them
echo "Downloading training images..."
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz 
mv train-images-idx3-ubyte.gz train-images.gz
gunzip train-images.gz

echo "Downloading training labels..."
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
mv train-labels-idx1-ubyte.gz train-labels.gz
gunzip train-labels.gz

echo "Downloading testing images..."
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
mv t10k-images-idx3-ubyte.gz test-images.gz
gunzip test-images.gz

echo "Downloading testing labels..."
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
mv t10k-labels-idx1-ubyte.gz test-labels.gz
gunzip test-labels.gz

exit 0

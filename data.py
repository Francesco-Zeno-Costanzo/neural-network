'''
file to convert the data from the mnist
dataset and collect them in two csv files
To download data:
wget https://raw.githubusercontent.com/fgnt/mnist/master/train-images-idx3-ubyte.gz
wget https://raw.githubusercontent.com/fgnt/mnist/master/train-labels-idx1-ubyte.gz
wget https://raw.githubusercontent.com/fgnt/mnist/master/t10k-images-idx3-ubyte.gz
wget https://raw.githubusercontent.com/fgnt/mnist/master/t10k-labels-idx1-ubyte.gz
then
gunzip *.gz
'''

def dataset(img_file, label_file, output_file, n):
    '''
    Parameters
    ----------
    img_file : str
        path of file with the images
    label_file : str
        path of file with the associated label
    output_file : str
        path of the csv file that can be read by other code
    '''

    f = open(img_file,   "rb")   # File with images
    l = open(label_file, "rb")   # File with labels
    o = open(output_file, "w")   # Output file
    
    # Remove the first 16 and 8 characters respectively
    f.read(16)
    l.read(8)
    
    images = [] # Will contain all data

    for i in range(n):
        # First column is target
        image = [ord(l.read(1))]
        # All the others are the pixel value from 0 to 255
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)
    
    # Save on file
    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    
    # Close files    
    f.close()
    l.close()
    o.close()


if __name__ == "__main__":
        
    # Train 
    path = 'MNIST_data/' 
    dataset(path + "train-images-idx3-ubyte",
            path + "train-labels-idx1-ubyte",
            path + "mnist_train.csv", 60000)
    # Test
    dataset(path + "t10k-images-idx3-ubyte",
            path + "t10k-labels-idx1-ubyte",
            path + "mnist_test.csv",  10000)

'''
file to convert the data from the mnist
dataset and collect them in two csv files
data from : http://yann.lecun.com/exdb/mnist/
'''

def dataset(img_file, label_file, output_file, n):

    f = open(img_file, "rb")   # file with images
    l = open(label_file, "rb") # file with labels
    o = open(output_file, "w") # outup file
    
    # remove the first 16 and 8 characters respectively
    f.read(16)
    l.read(8)
    
    images = [] # will contain all data

    for i in range(n):
        # first collum is target
        image = [ord(l.read(1))]
        # alle the others are the pixel value from 0 to 255
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)
    
    # save on file
    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    
    # close file    
    f.close()
    l.close()
    o.close()
    
# Train 
path = 'MNIST_data/' 
dataset(path + "train-images-idx3-ubyte",
        path + "train-labels-idx1-ubyte",
        path + "mnist_train.csv", 60000)
# Test
dataset(path + "t10k-images-idx3-ubyte",
        path + "t10k-labels-idx1-ubyte",
        path + "mnist_test.csv",  10000)

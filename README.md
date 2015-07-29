# cifar.py

Python library for handling the CIFAR datasets [1]

[1] http://www.cs.toronto.edu/~kriz/cifar.html

### Docs:
```
# get_images: get images from binary cpython pickles
# blobs: array of paths to binary cpython pickles
# label names: array of label names in which indices corresponds to labels' numbers in CIFAR (get from get_label_names)
# returns: array of objects with type Image
def get_images(blobs, label_names)

# get_label_names: get label names from label names cpython binary file
# blob: path to label names cpython binary file (generally matches /meta/ in the filename)
# returns: array of strings in which indices of label names correspond to labels' numbers in CIFAR
def get_label_names(blob)

# CIFAR_Image: Class for containing images
# name: Name of image
# data: 1 * 3072 array of rgb values from the cifar dataset
# mat_mode: If should get matrices for rgb values (or don't with a false value)
# members: 
	# self.name: name of image
	# self.data: 1 * 3072 array of rgb values from the cifar dataset
	# self.red_mat or self.blue_mat or self.green_mat: 32 * 32 array of arrays of color intensity values corresponding to pixel locations
		# ex: self.blue_mat[10][25] is the intensity of the pixel at row 10 column 25 of the CIFAR image

class CIFAR_Image:
    def __init__(self, name, data, mat_mode=True):
        self.data = data
        self.name = name
        if mat_mode:
            self.red_mat = self.color_matrix(0, 32, 32)
            self.green_mat = self.color_matrix(1024, 32, 32)
            self.blue_mat = self.color_matrix(2048, 32, 32)
```

### Example:
```
from cifar import *

blobs = ["cifar-10-batches-py/data_batch_1","cifar-10-batches-py/data_batch_2","cifar-10-batches-py/data_batch_3", "cifar-10-batches-py/data_batch_4","cifar-10-batches-py/data_batch_5"]
label_names = get_label_names("cifar-10-batches-py/batches.meta")
images = get_images(blobs, label_names)
```

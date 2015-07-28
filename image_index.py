# image indexer

import cPickle
import numpy as np
import math, os, pdb

def un_pickle(file_path):
    with open(file_path, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

class CIFAR_Image:
    def __init__(self, name, data, mat_mode=True):
        self.data = data
        self.name = name
        if mat_mode:
            self.red_mat = self.color_matrix(0, 32, 32)
            self.green_mat = self.color_matrix(1024, 32, 32)
            self.blue_mat = self.color_matrix(2048, 32, 32)
            del self.data

    def color_matrix(self, start_index, num_rows, num_columns):
        mat = [[None] * num_columns] * num_rows
        num_elements = num_rows * num_columns
        for i in range(num_elements):
            mat[i/num_columns][i % num_rows] = self.data[i + start_index]
        return mat
            
def get_images(blobs, label_names):
    images = []
    for blob in blobs:
        blob_dict = un_pickle(blob)
        data = blob_dict['data']
        labels = blob_dict['labels']
        for i in range(len(labels)):
            images.append(CIFAR_Image(label_names[labels[i]], data[i]))
            if i % 100 == 0:
                print str(int(100 * float(len(images))/50000)) + "%"
    return images

def get_label_names(blob):
    label_dict = un_pickle(blob)['label_names']
    return label_dict

def pickle(obj, file_path):
    with open(file_path, "wb") as fo:
        cPickle.dump(obj, fo)

if __name__ == "__main__":
    if os.path.exists('images.peck'):
        images = un_pickle('images.peck')
    else:
        blobs = ["/cluster/logan/scraped_data/cifar/cifar-10-batches-py/data_batch_1","/cluster/logan/scraped_data/cifar/cifar-10-batches-py/data_batch_2","/cluster/logan/scraped_data/cifar/cifar-10-batches-py/data_batch_3", "/cluster/logan/scraped_data/cifar/cifar-10-batches-py/data_batch_4","/cluster/logan/scraped_data/cifar/cifar-10-batches-py/data_batch_5"]
        label_names = get_label_names("/cluster/logan/scraped_data/cifar/cifar-10-batches-py/batches.meta")
        images = get_images(blobs, label_names)
    pickle(images, "images.peck")

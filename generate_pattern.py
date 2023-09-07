from skimage import io, color
from skimage.transform import rescale, downscale_local_mean
from sklearn.cluster import KMeans
import numpy as np
from thread_loader import load_threads
import argparse

parser = argparse.ArgumentParser(
                    prog='Embroidery Thread Finder',
                    description='Simplifies an image and outputs DMC thread values')

parser.add_argument('-c', '--count', default=16, type=int)  
parser.add_argument('-f', '--filename', default="./yosemite-valley.jpg")  

args = parser.parse_args()

print(args.count)

threads = load_threads()

image = io.imread(args.filename)

image_smaller = rescale(image, 0.25, anti_aliasing=True)
image_downscaled = downscale_local_mean(image, (10, 10,10))


image = color.rgb2lab(image)

#Dimension of the original image
rows = image.shape[0]
cols = image.shape[1]


#Flatten the image
image = image.reshape(rows*cols, 3)


k = args.count
kmeans = KMeans(n_clusters=k)   
kmeans.fit(image)

#Replace each pixel value with its nearby centroid
compressed_image = kmeans.cluster_centers_[kmeans.labels_]

#Reshape the image to original dimension
compressed_image = compressed_image.reshape(rows, cols, 3)

#convert image back to RGB colour space
compressed_image = color.lab2rgb(compressed_image)


# Get all unique colours in the image
unique_colours = np.unique(compressed_image.reshape(-1, compressed_image.shape[2]), axis=0)
#map the unique colours to their usual 0-255 value
mapped_colours = []
for col in unique_colours:
    colour = []
    for comp in col:
        colour.append(int(comp * 255))
    mapped_colours.append(colour)



#Find closest thread:
#currently uses RGB value -> needs to be converted to LAB
mapped_threads = []
for col in mapped_colours:
    closest_thread = threads[0]
    smallest_current_distance = -1
    for thread in threads:
        euc_dist = (thread.rgb[0] - col[0])**2 + (thread.rgb[1] - col[1])**2 + (thread.rgb[2] - col[2])**2
        if smallest_current_distance < 0:
            smallest_current_distance = euc_dist
            closest_thread = thread
        else:
            if euc_dist < smallest_current_distance:
                smallest_current_distance = euc_dist
                closest_thread = thread
    mapped_threads.append(closest_thread)

print("\nDMC Threads to use for image")
for th in mapped_threads:
    print(th)


io.imshow(compressed_image)
io.show()
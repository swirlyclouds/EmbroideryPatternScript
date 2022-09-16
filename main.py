
from skimage import io, color
from sklearn.cluster import KMeans
import numpy as np
import csv

#Read the image
image = io.imread('.\Test_Image (2).jpg')

# convert image from RGB to LAB colour space
# LAB is perceptually uniform and allows for better finding of closest colours that visually look similar
# rather than colours that are just similar by their RGB value which can be inaccurate
image = color.rgb2lab(image)

#Dimension of the original image
rows = image.shape[0]
cols = image.shape[1]

#Flatten the image
image = image.reshape(rows*cols, 3)
#Implement k-means clustering to form k clusters
k = 16
kmeans = KMeans(n_clusters=k)
kmeans.fit(image)

#Replace each pixel value with its nearby centroid
compressed_image = kmeans.cluster_centers_[kmeans.labels_]

#Reshape the image to original dimension
compressed_image = compressed_image.reshape(rows, cols, 3)

#convert image back to RGB colour space
compressed_image = color.lab2rgb(compressed_image)

#Save and display output image
io.imsave(f'compressed_image_{k}.jpg', compressed_image)
io.imshow(compressed_image)
#io.show()

# Get all unique colours in the image
unique_colours = np.unique(compressed_image.reshape(-1, compressed_image.shape[2]), axis=0)

#map the unique colours to their usual 0-255 value
mapped_colours = []
for col in unique_colours:
    colour = []
    for comp in col:
        colour.append(int(comp * 255))
    mapped_colours.append(colour)
    
# Get DMC Threads
class Thread:
    def __init__(self, DMCcode, coloursName, hexValue):
        self.code = DMCcode
        self.name = coloursName
        h = hexValue.lstrip('#')
        self.rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    
    def __str__(self):
        return f"DMC: {self.code}\t Name: {self.name}\t\t\t\t RGB: {self.rgb}"

threads = []
with open('DMC_threads.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        threads.append(Thread(row[0], row[1], row[2][1:]))

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

io.show()
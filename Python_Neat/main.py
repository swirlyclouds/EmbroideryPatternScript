from skimage import io, color, draw
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math
import csv

# Get DMC Threads
class Thread:
    def __init__(self, DMCcode, coloursName, hexValue):
        self.code = DMCcode
        self.name = coloursName
        h = hexValue.lstrip('#')
        self.rgb = tuple(int(h[i:i+2], 16)/255 for i in (0, 2, 4))
        self.lab = convertToLAB(self.rgb)
    
    def __str__(self):
        return f"DMC: {self.code} \t RGB: {self.rgb} \t LAB: {self.lab}\t Name: {self.name}"

threads = []

# 1. import image
# 2. import threads
# 3. simplify image
# 4. reduce noise

def get_image(fname):
    #Read the image
    image = io.imread(fname)

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
    compressed_image = kmeans.cluster_centers_[kmeans.labels_] #/ 255

    #Reshape the image to original dimension
    compressed_image = compressed_image.reshape(rows, cols, 3)

    io.imshow(color.lab2rgb(compressed_image))
    return compressed_image

def load_threads():
    with open('.\..\DMC_threads.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile)
        i = 0
        for row in spamreader:
            if i < 1:
                i = 1
            else:
                x = Thread(row[0], row[1], row[2][1:])
                threads.append(x)

def convertToLAB(rgb):
    r = rgb[0]
    g = rgb[1]
    b = rgb[2]

    r = pow((r + 0.055)/ 1.055, 2.4) if (r > 0.04045)  else r / 12.92
    g = pow((g + 0.055)/ 1.055, 2.4) if (g > 0.04045)  else g / 12.92
    b = pow((b + 0.055)/ 1.055, 2.4) if (b > 0.04045)  else b / 12.92

    x = (r * 0.4124 + g * 0.3576 + b * 0.1805) / 0.95047
    y = (r * 0.2126 + g * 0.7152 + b * 0.0722) / 1.00000
    z = (r * 0.0193 + g * 0.1192 + b * 0.9505) / 1.08883

    x = pow(x, 1/3) if (x > 0.008856) else (7.787 * x) + 16/116
    y = pow(y, 1/3) if (y > 0.008856) else (7.787 * y) + 16/116
    z = pow(z, 1/3) if (z > 0.008856) else (7.787 * z) + 16/116

    return ((116 * y) - 16, 500 * (x - y), 200 * (y - z))  

def map_threads(mapped_colours):
    #Find closest thread:
    #currently uses RGB value -> needs to be converted to LAB
    mapped_threads = []
    for col in mapped_colours:
        closest_thread = threads[0]
        smallest_current_distance = -1
        for thread in threads:
            #euc_dist = (thread.rgb[0] - col[0])**2 + (thread.rgb[1] - col[1])**2 + (thread.rgb[2] - col[2])**2
            euc_dist = math.sqrt((thread.lab[1] - col[1])**2 + (thread.lab[2] - col[2])**2)  + abs(thread.lab[0] - col[0])
            if smallest_current_distance < 0:
                smallest_current_distance = euc_dist
                closest_thread = thread
            else:
                if euc_dist < smallest_current_distance:
                    smallest_current_distance = euc_dist
                    closest_thread = thread
        mapped_threads.append(closest_thread)
    return mapped_threads


if __name__ == "__main__":
    load_threads()
    
    img_means_lab = get_image(".\..\CoL_Img_1.jpg")      
        
    # Get all unique colours in the image
    unique_colours = np.unique(img_means_lab.reshape(-1, img_means_lab.shape[2]), axis=0) 

    mapped_colours = []
    for col in unique_colours:
        colour = []
        for comp in col:
            colour.append((comp))
        mapped_colours.append(colour)     

    m_threads = map_threads(mapped_colours)
    image = img_means_lab.copy()
    image = color.lab2rgb(image)
    for t_i in range(len(m_threads)):
        row, col = draw.rectangle(start=(image.shape[0] - 50, int(t_i * (image.shape[1] - 1) / (len(m_threads) + 1))), end=(image.shape[0] - 1,  int((t_i + 1) * (image.shape[1] - 1) / (len(m_threads) + 1)))) # image.shape[0]
        #row, col = draw.rectangle(start=(150,150), end=(175, 200))
        print(m_threads[t_i])
        #plt.gca().add_patch(patches.Rectangle((image.shape[0] - 50, int(t_i * (image.shape[1] - 1) / (len(m_threads) + 1))), 40,(image.shape[1] - 1) / (len(m_threads) + 1),linewidth=1,edgecolor='none',facecolor=[m_threads[t_i].rgb[0],m_threads[t_i].rgb[1],m_threads[t_i].rgb[2]] ))
       # io.text(row, col, "some text", dict(color='blue', va='center', ha='center'))
        #image[row, col, :] = [255, 255, 0]
        image[row, col, :] = [m_threads[t_i].rgb[0],m_threads[t_i].rgb[1],m_threads[t_i].rgb[2]] 
    plt.imshow(image)
    plt.show()

    
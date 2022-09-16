# EmbroideryPatternScript
 Takes an image, simplifies the colours, generates a list of DMC embroidery threads that could be used to make the image as an embroidered piece  of fabric


## How it works
### Part 1: Simplyifying the image
The image is loaded into the script as an array of pixel values (R,G,B)

RGB is then converted into LAB colourspace. This is so that colours found during the next step are more perceptually similar to one another than would be the case if using RGB.

K-means Clustering is then used to simplify the image into just 16 unique colours. 

### Part 2: Finding the threads
Load the DMC thread data from the csv file into a list of custom classes.

Iterate through the colours in the image to find the closest thread using the euclidean distance of the colour values.

Currently done through RGB, will be changed to LAB for reasons stated above. 

## How to use:
Add an image into the same directory as the script with the name "Test_Image.jpg" 

Will be changed to have the filename passed in as an argument

##Example Output: 
![image](https://user-images.githubusercontent.com/30084184/190827073-482989a7-ef64-4414-bc03-6f8bb64ae14c.png)

#import packages
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

#stores images in the output_images_task_2 directory
path = "output_images_task_2"
if not(os.path.isdir(path)):
    os.mkdir(path)

#Task A :
#load image 1, convert to grayscale, change type to 'float32' and display the original and grayscale images.
input_image_1 = cv2.imread("Assignment_MV_01_image_1.jpg")
cv2.imshow("Image_1", input_image_1)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(path+"/Image_1.jpg", input_image_1)
image_1 = cv2.cvtColor(input_image_1, cv2.COLOR_RGB2GRAY)
cv2.imshow("Image_1_gray", image_1)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(path+"/Image_1_gray.jpg", image_1)
image_1 = image_1.astype(np.float32)

#load image 2, convert to grayscale, change type to 'float32' and display the original and grayscale images.
input_image_2 = cv2.imread("Assignment_MV_01_image_2.jpg")
cv2.imshow("Image_2", input_image_2)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(path+"/Image_2.jpg", input_image_2)
image_2 = cv2.cvtColor(input_image_2, cv2.COLOR_RGB2GRAY)
cv2.imshow("Image_2_gray", image_2)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(path+"/Image_2_gray.jpg", image_2)
image_2 = image_2.astype(np.float32)


#Task B :
#draw a rectangle around the window with the given co-ordinates on image 1 and display the image
cv2.rectangle(input_image_1,(360,210), (430,300),(0, 255, 0), 2)
cv2.imshow("Image_1_rectangle",input_image_1)
cv2.imwrite(path+"/Image_1_rectangle.jpg", input_image_1)

#crop the image along the rectangle and display the cropped image.
cropped_image = input_image_1[210:300, 360:430]
cv2.imshow("cropped", cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(path+"/cropped.jpg", cropped_image)

#convert the cropped image to grayscale and change the data type to 'float32' for further processing
cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2GRAY)
cropped_image = cropped_image.astype(np.float32)

#Task C :
#calculate the mean and standard deviation of the cropped image(cut-out patch) from image 1
cropped_image_mean = np.mean(cropped_image)
cropped_image_std = np.std(cropped_image)
height,width = cropped_image.shape

#use this 'for' loop to traverse through the entire image 2.
cross_correlation = []
images = []
for i in range(len(image_2)- height):
    for j in range(len(image_2[0])-width):

        #cut out patch of equal size as the cropped image
        patch = image_2[i:i+height, j:j+width]

        #calculate mean and standard deviation of this patch
        patch_mean = np.mean(patch)
        patch_std = np.std(patch)

        #calculate cross-correlation between the two patches
        num = np.sum((cropped_image - cropped_image_mean) * (patch-patch_mean))
        den = cropped_image_std * patch_std
        ccr = num/den
        cross_correlation.append(ccr)
        images.append(((j,i),(j+width,i+height)))


#identify the patch with the maximum cross-correlation on image 2
max_ccr = max(cross_correlation)
max_ccr_patch = images[cross_correlation.index(max_ccr)]

#draw a rectangle around the identified patch and display the image
cv2.rectangle(input_image_2,max_ccr_patch[0],max_ccr_patch[1],(0, 255, 0), 2)
cv2.imshow("Output Image - 2",input_image_2)
cv2.imwrite(path+"/Output Image - 2.jpg", input_image_2)


cv2.waitKey(0)
cv2.destroyAllWindows()

#import packages
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os

#stores images in the output_images_task_1 directory
path = "output_images_task_1"
if not(os.path.isdir(path)):
    os.mkdir(path)

#function to identify the keypoints in each of the DoG images
def non_maximum_suppression(img,T,sigma,previous,next):
    points = []
    for x in range(1, len(img) - 1):
        for y in range(1, len(img[0]) - 1):
            #'if' condidtion checking whether a particular point in the DoG image has higher values than its surrounding 26 points.
            if ((img[x,y]>T) and
                (img[x,y]>img[x-1,y-1]) and
                (img[x,y]>img[x-1,y]) and
                (img[x,y]>img[x-1,y+1]) and
                (img[x,y]>img[x,y-1]) and
                (img[x,y]>img[x,y+1]) and
                (img[x,y]>img[x+1,y-1]) and
                (img[x,y]>img[x+1,y]) and
                (img[x,y]>img[x+1,y+1]) and
                (img[x,y]>previous[x,y]) and
                (img[x, y] > previous[x - 1, y - 1]) and
                (img[x, y] > previous[x - 1, y]) and
                (img[x, y] > previous[x - 1, y + 1]) and
                (img[x, y] > previous[x, y - 1]) and
                (img[x, y] > previous[x, y + 1]) and
                (img[x, y] > previous[x + 1, y - 1]) and
                (img[x, y] > previous[x + 1, y]) and
                (img[x, y] > previous[x + 1, y + 1]) and
                (img[x, y] > next[x, y]) and
                (img[x, y] > next[x - 1, y - 1]) and
                (img[x, y] > next[x - 1, y]) and
                (img[x, y] > next[x - 1, y + 1]) and
                (img[x, y] > next[x, y - 1]) and
                (img[x, y] > next[x, y + 1]) and
                (img[x, y] > next[x + 1, y - 1]) and
                (img[x, y] > next[x + 1, y]) and
                (img[x, y] > next[x + 1, y + 1])):
                points.append((x,y,sigma))
    return points

#function to calculate the best orientation angle for a key point using histogram of orientation gradients
def hog(gradient_length, gradient_direction, weighting_function):

    #create a list of 36 bins representing angles from 0 to 360 degrees.
    bin_values = [0] * 36
    weighted_gradient_lengths = {}

    #calculate the weighted_gradient_lengths
    for key1,value1 in gradient_length.items():
        for key2,value2 in weighting_function.items():
            if (key1 == key2):
                weighted_gradient_lengths[key1] = value1*value2

    #if the gradient direction of a point in the 7X7 grid of a keypoint falls in the range of 0-9, 10-19, 20-29.....350-359, then the weighted gradient length will be added to the (int(angle/10))th bin in the list.
    #eg : if the gradient direction is 45 degrees, then the weighted gradient length will be added to int(45/10) = 4th bin.
    for key,value in gradient_direction.items():
        bin_values[int(abs(gradient_direction[key])/10)] += weighted_gradient_lengths[key]

    #the bin having the highest magnitude will be chosen as the best orientation angle
    best_orientation_angle = bin_values.index(max(bin_values)) *10
    return best_orientation_angle

#function to draw circles and indicate the orientation angles around the keypoints on the original input image
def draw_key_point_circles(list_keypoints, input_image):
    for i in list_keypoints:
        for point,orientation in i.items():

            #draw circle around the keypoint
            cv2.circle(input_image, ( int(point[1]/2),int(point[0]/2)), 3 * int(point[2]/2), (0, 255, 0), 2)

            #draw a line connecting the centre of the cirle and the circumference with radius : 3*sigma and angle : best orientation angle.
            cv2.line(input_image, ( int(point[1]/2),int(point[0]/2)), (int(int(point[1]/2) + 3 *int(point[2]/2)* np.cos(abs(orientation))), int(int(point[0]/2) + 3 *int(point[2]/2)*np.sin(abs(orientation)))),(0, 255, 0), 2)

    #display the output image with all the keypoints
    WINDOW_NAME = "Output_image "
    cv2.namedWindow(WINDOW_NAME)
    cv2.startWindowThread()
    cv2.imshow(WINDOW_NAME, input_image)
    cv2.imwrite(path + "/Output_image.jpg", input_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



#Task A : Load image, convert to gray channel, convert data type to 'float32', determine the size of the image and double its size.
input_image = cv2.imread("Assignment_MV_01_image_1.jpg")
cv2.imshow("input", input_image)
cv2.imwrite(path + "/Input.jpg", input_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#convert image to gray-scale
img = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
cv2.imshow("gray", img)
cv2.imwrite(path + "/gray.jpg", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#convert data type to 'float32'
img = img.astype(np.float32)

#display current size of image
height,width = img.shape
print("Current image size: ",height,width)

#double the size of the image
img = cv2.resize(img,(width*2,height*2))
resized_input_image = img
height,width = img.shape
print("Image size after doubling : ",height,width)


#Task B : Create 12 Guassian smoothing kernels, plot them as images, apply these kernels to the input image to create 12 scal-space representation of the input image
scale_space_images = []

for k in range(12):

    #generate the 12 Guassian smoothing kernels
    sigma = 2**(k/2)
    x, y = np.meshgrid(np.arange(-3*sigma, 3*sigma), np.arange(-3*sigma, 3*sigma))
    kernel = (1/(2*np.pi*sigma**2))* (np.exp(-((x**2+y**2)/(2*sigma**2))))

    #plot the kernels as images
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x,y,kernel)
    fig.show()

    #apply the kernels to the input image to obtain scale space representations
    img = cv2.filter2D(img, -1, kernel)
    scale_space_images.append(img)

    #display the smoothened images
    WINDOW_NAME = "Image - Guassian Smoothing " + str(k)
    cv2.namedWindow(WINDOW_NAME)
    cv2.startWindowThread()
    cv2.imshow(WINDOW_NAME, img / np.max(img))
    cv2.imwrite(path + "/Image - Guassian Smoothing "+str(k)+".jpg", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#Task C : calculate and display DoG images
#generate the Difference of Guassian images (11 images)
DoG = []
for i in range(len(scale_space_images)-1):
    DoG.append(scale_space_images[i]-scale_space_images[i+1])


key_points = {}
for i in range(len(DoG)):
    #display the DoG images
    img = DoG[i]
    WINDOW_NAME = "DoG Image " + str(i)
    cv2.namedWindow(WINDOW_NAME)
    cv2.startWindowThread()
    cv2.imshow(WINDOW_NAME, img)
    cv2.imwrite(path + "/DoG Image " + str(i) + ".jpg", img/ np.max(img))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #Task D : Identify the keypoints in the DoG images using threshold =10
    T = 10
    sigma = 2**(i/2)
    #keypoints will not be calculated for the first and last DoG images.
    if i != 0 and i != len(DoG)-1:
        key_points[i] = (non_maximum_suppression(DoG[i],T,sigma,DoG[i-1],DoG[i+1]))

#Task E : calculate the x and y derivates of the scale space representations(24 images) and display them
fx = {}
fy = {}
for i in range(len(scale_space_images)):
    #dx and dy kernels
    dx = np.array([[1,0,-1]])
    dy = np.array([[1],[0],[-1]])

    fx[i] = cv2.filter2D(scale_space_images[i], -1, dx)
    fy[i] = cv2.filter2D(scale_space_images[i], -1, dy)

    #display the images
    WINDOW_NAME = "Image - Derivatives(dx) " + str(i)
    cv2.namedWindow(WINDOW_NAME)
    cv2.startWindowThread()
    cv2.imshow(WINDOW_NAME, fx[i] / np.max(fx[i]))
    cv2.imwrite(path + "/Image - Derivatives(dx) " + str(i) + ".jpg", fx[i]/np.max(fx[i]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    WINDOW_NAME = "Image - Derivatives(dy) " + str(i)
    cv2.namedWindow(WINDOW_NAME)
    cv2.startWindowThread()
    cv2.imshow(WINDOW_NAME, fy[i] / np.max(fy[i]))
    cv2.imwrite(path + "/Image - Derivatives(dy) " + str(i) + ".jpg", fy[i] / np.max(fy[i]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Task F : calculate gradient length, gradient direction and weighting function. Also, find the best orientation angle using the HOG method.
list_keypoints = []

#for every DoG image,
for DoG_image_no,key_points in key_points.items():
    key_point_orientation = {}

    for points in key_points:
        x = points[0]
        y = points[1]
        sigma = points[2]

        # for every key point, we identify a 7X7 grid of points
        q = np.array([])
        r = np.array([])
        for k in range(-3,3):
            q = np.append(q,x+(3/2*k*sigma))
            r = np.append(r,y+(3/2*k*sigma))
        q,r = np.meshgrid(q,r)

        gradient_length = {}
        gradient_direction = {}
        weighting_function = {}
        for i in range(len(q)):
            for j in range(len(q[0])):

                #for every point in the 7X7 grid,
                x = int(q[i][j])
                y = int(r[i][j])

                #check whether the point is within the size of the image
                if x < 1536 and x > -1 and y < 2048 and y > -1 :

                    #identify the X and Y derivatives of the image corresponding to the correct scale space(sigma)
                    x_derivative = fx[DoG_image_no]
                    y_derivative = fy[DoG_image_no]

                    #identify the values at the co-ordinates from the X and Y derivative images
                    gx = x_derivative[x][y]
                    gy = y_derivative[x][y]

                    #calculate the gradient length
                    gradient_length[(x,y)] = np.sqrt((gx**2+gy**2))

                    #calculate the gradient direction : if the angle is negative, do 360-angle.
                    gradient_direction[(x,y)] = math.degrees(math.atan((gx/gy)))
                    if gradient_direction[(x,y)] <0:
                        gradient_direction[(x,y)] = 360 + gradient_direction[(x,y)]

                    #calculate the weighting function
                    weighting_function[(x,y)] = np.exp(-((x**2+y**2)/((9*sigma**2)/2)))/((9*np.pi*sigma**2)/2)

        #calculate the best orientation angle for the keypoint using HOG and store it in a dictionary
        key_point_orientation[points] = hog(gradient_length, gradient_direction, weighting_function)

    #append all the keypoints data of every DoG image into the list
    list_keypoints.append(key_point_orientation)

#Task G : represent the keypoints on the original input image
#represent the keypoints on the input image.
draw_key_point_circles(list_keypoints,input_image)



cv2.waitKey(0)
cv2.destroyAllWindows()


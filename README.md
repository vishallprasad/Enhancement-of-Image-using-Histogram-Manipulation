# Enhancement of Image using Histogram Manipulation

## Abstract
- Image enhancement is the process of adjusting digital images so that the results are more suitable for display or further image analysis. Image enhancement can be done by Histogram equalization. Histogram equalization is a technique for adjusting image intensities to enhance contrast. 
- Digital Image Processing is a rapidly evolving field with the growing applications in science & engineering. Image Processing holds the possibility of developing an ultimate machine that could perform visual functions of all living beings. 
- The image processing is a visual task, the foremost step is to obtain an image i.e. image acquisition then enhancement and finally to process.
- Image enhancement is basically improving the digital image quality. Image histogram is helpful in image enhancement. The histogram in the context of image processing is the operation by which the occurrences of each intensity value in the image is shown and Histogram equalization is the technique by which the dynamic range of the histogram of an image is increased

## Objective
- A histogram is a very important tool in Image processing. It is a graphical representation of the distribution of data. An image histogram gives a graphical representation of the distribution of pixel intensities in a digital image.
- The x-axis indicates the range of values the variable can take. This range can be divided into a series of intervals called bins. The y-axis shows the count of how many values fall within that interval or bin.

→ When plotting the histogram we have the pixel intensity in the X-axis and the frequency in the Y-axis. As any other histogram we can decide how many bins to use.
- A histogram can be calculated both for the gray-scale image and for the colored image. In the first case we have a single channel, hence a single histogram. In the second case we have 3 channels, hence 3 histograms.
- Calculating the histogram of an image is very useful as it gives an intuition regarding some properties of the image such as the tonal range, the contrast and the brightness.

→ To identify the dominant colors in an image, we can use the histogram plot of the Hue channel.
- In an image histogram, the x-axis represents the different color values, which lie between 0 and 255, and the y-axis represents the number of times a particular intensity value occurs in the image.

## Methodology
#### Calculating the Histogram

OpenCV provides the function cv2.calcHist to calculate the histogram of an image. The signature is the following :
```ruby
cv2.calcHist(images, channels, mask, bins, ranges)
```
where :

**images -** is the image we want to calculate the histogram of wrapped as a list, so if our image is in variable image we will pass [image],

**channels -** is the the index of the channels to consider wrapped as a list ([0] for gray-scale images as there's only one channel and [0], [1] or [2] for color images if we want to consider the channel green, blue or red respectively),

**mask -** is a mask to be applied on the image if we want to consider only a specific region (we're gonna ignore this in this post),

**bins -** is a list containing the number of bins to use for each channel,

**ranges -** is the range of the possible pixel values which is [0, 256] in case of RGB color space (where 256 is not inclusive).

The returned value hist is a numpy.ndarray with shape (n_bins, 1) where hist[i][0] is the number of pixels having an intensity value in the range of the i-th bin.

We can simplify this interface by wrapping it with a function that in addition to calculate the histogram it also draws it (at the moment we’re going to fix the number of bins to 256) :

## Gray-scale histogram

Here is a simple code for just loading the image :
```ruby
import cv2
import numpy as np

gray_img = cv2.imread('Elon Musk.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('RICHEST MAN',gray_img)

while True:
    k = cv2.waitKey(0) & 0xFF
    if k == 27: break             # ESC key to exit
cv2.destroyAllWindows()
```
**How GrayScale Image looks like :**
| :--: |
![Untitled 1](https://user-images.githubusercontent.com/77826589/118615725-a394e900-b7de-11eb-9632-b75fd1bf373f.png)

**Plotting histogram for a gray-scale image.**
| :--: |
![image (1)](https://user-images.githubusercontent.com/77826589/118624032-6cc2d100-b7e6-11eb-9980-fe15c7f585bc.png)
#### The code for histogram looks like this :
```ruby
import cv2
import numpy as np
from matplotlib import pyplot as plt

gray_img = cv2.imread('Elon Musk.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('RICHEST MAN',gray_img)
hist = cv2.calcHist([gray_img],[0],None,[256],[0,256])
plt.hist(gray_img.ravel(),256,[0,256])
plt.title('Histogram for gray scale picture')
plt.show()

while True:
    k = cv2.waitKey(0) & 0xFF
    if k == 27: break             # ESC key to exit
cv2.destroyAllWindows()
```
## NumPy - np.histogram()
**NumPy** also provides us a function for histogram, **np.histogram()**. So, we can use NumPy fucntion instead of OpenCV function :

```ruby
import cv2
import numpy as np
from matplotlib import pyplot as plt

gray_img = cv2.imread('Elon Musk.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('RICHEST MAN',gray_img)
#hist = cv2.calcHist([gray_img],[0],None,[256],[0,256])
hist,bins = np.histogram(gray_img,256,[0,256])

plt.hist(gray_img.ravel(),256,[0,256])
plt.title('Histogram for gray scale picture')
plt.show()

while True:
    k = cv2.waitKey(0) & 0xFF
    if k == 27: break             # ESC key to exit
cv2.destroyAllWindows()
```

## Histogram of color image
**Let's draw RGB histogram :**
![image](https://user-images.githubusercontent.com/77826589/118604276-9c67de00-b7d2-11eb-9f02-2b578fe1b8e1.jpg)

**Plotting Histogram for a color image.**
| :--: |
![image (1)](https://user-images.githubusercontent.com/77826589/118626538-954bca80-b7e8-11eb-86b1-20737d5025e6.jpg)
#### The code of RGB Histogram :
```ruby
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('Elon Musk.jpg', -1)
cv2.imshow('GoldenGate',img)

color = ('b','g','r')
for channel,col in enumerate(color):
    histr = cv2.calcHist([img],[channel],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.title('Histogram for color scale picture')
plt.show()

while True:
    k = cv2.waitKey(0) & 0xFF
    if k == 27: break             # ESC key to exit
cv2.destroyAllWindows()
```

## Histogram Equalization
- The histogram equalization process is an image processing method to adjust the contrast of an image by modifying the image’s histogram.
- The intuition behind this process is that histograms with large peaks correspond to images with low contrast where the background and the foreground are both dark or both light. Hence histogram equalization stretches the peak across the whole range of values leading to an improvement in the global contrast of an image.
- It is usually applied to gray-scale images and it tends to produce unrealistic effects, but it is highly used where a high contrast is needed such as in medical or satellite images.
                                                                                                      
| ![Picture1](https://user-images.githubusercontent.com/77826589/118763311-4019c280-b895-11eb-899c-2ec8baf8a35a.png) |
| :--: |
**Histograms of an image before and after equalization**

OpenCV provides the function cv2.equalizeHist to equalize the histogram of an image. The signature is the following :
```ruby
cv2.equalizeHist(image)
```

I would recommend you to read the wikipedia page on [Histogram Equalization](https://en.wikipedia.org/wiki/Histogram_equalization) for more details about it. It has a very good explanation with worked out examples, so that you would understand almost everything after reading that. Instead, here we will see its Numpy implementation. After that, we will see OpenCV function.

```ruby
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread("Taj Hotel.jpg",0)
hist,bins = np.histogram(img.flatten(),256,[0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * float(hist.max()) / cdf.max()
plt.plot(cdf_normalized, color = 'b')
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()
```
**Before Histogram Equalization**
| :--: |
![image (4)](https://user-images.githubusercontent.com/77826589/118650182-ca184b80-b801-11eb-92a0-32edf03704b6.jpg)

**After Histogram Equalization**
| :--: |
![image (5)](https://user-images.githubusercontent.com/77826589/118759405-9b948200-b88e-11eb-9bab-407911a1fefb.jpg)

## Histograms Equalization in OpenCV
OpenCV has a function to do this, **cv.equalizeHist()**. Its input is just grayscale image and output is our histogram equalized image.
Below is a simple code snippet showing its usage for same image we used :

```ruby
import cv2
import numpy as np
img = cv2.imread('Taj Hotel.jpg', 0)
equ = cv2.equalizeHist(img)
res = np.hstack((img, equ))
cv2.imshow('Taj Hotel', res)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

![taj hotel ](https://user-images.githubusercontent.com/77826589/118652589-60e60780-b804-11eb-8fa4-943dcd4da8d3.png)
So now you can take different images with different light conditions, equalize it and check the results.

Histogram equalization is good when histogram of the image is confined to a particular region. It won't work good in places where there is large intensity variations where histogram covers a large region, ie both bright and dark pixels are present.

## References
http://en.wikipedia.org/wiki/Histogram_equalization

https://docs.opencv.org/master/d5/daf/tutorial_py_histogram_equalization.html

import cv2
import matplotlib.pyplot as plt

# Load the images
image1 = cv2.imread(r'F:\linshi\B007_cut.png')
image2 = cv2.imread(r'F:\linshi\L007_cut.png')

# Resize the images to the same size
image1_resized = cv2.resize(image1, (100, 100))
image2_resized = cv2.resize(image2, (100, 100))

# Calculate the histogram for the resized images
hist1 = cv2.calcHist([image1_resized], [0], None, [256], [5,256])
hist2 = cv2.calcHist([image2_resized], [0], None, [256], [5,256])

# Plot the histogram
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(hist1)
plt.title('Histogram for image1')
plt.subplot(1, 2, 2)
plt.plot(hist2)
plt.title('Histogram for image2')
plt.show()

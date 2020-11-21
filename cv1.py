import cv2
image = cv2.imread("/home/User/Pictures/multimedia2.png")
print(image.shape)
cv2.imshow("Image" , image)
cv2.waitKey(0)


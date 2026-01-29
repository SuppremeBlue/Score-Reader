import cv2 as cv
import sys

image = "debussy_1"

img = cv.imread(f"{image}.png")

if img is None:
    sys.exit("Unable to read image")

cv.imshow("Display Window", img)
k = cv.waitKey(0)
if k == ord("s"):
    cv.imwrite(f"processed_{image}.png",img)
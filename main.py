from Detector import *

detector = Detector()



import glob, os
os.chdir("new_images/")
for file in glob.glob("*.jpg"):
    print(file)
    detector.onImage(file)
    pause()

# detector.onImage("1.jpg")
# detector.onImage("2.png")
# detector.onVideo()

print("Done.")

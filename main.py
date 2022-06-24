from Detector import *
detector = Detector()

#%%

'''
Merge two json files as follows:
Train:
# python3 merge_train.py train1.json train2.json train_dataset.json
Test:
# python3 merge_test.py test1.json test2.json test_dataset.json
'''

#%%

import time
import glob, os
os.chdir("new_images/")
for file in glob.glob("*.jpg"):
    print(file)
    detector.onImage(file)
    time.sleep(3)
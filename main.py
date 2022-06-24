from Detector import *
detector = Detector()

#%%

'''
Merge two json files as follows:
Train:
# python3 merge.py dataset1.json dataset2.json total_dataset.json
'''

'''
Divide the dataset to two json files (Train vs Test) as follows:

# python3 split.py --having-annotations --multi-class -s 0.7 total_dataset.json train.json test.json
'''

#%%

import time
import glob, os
os.chdir("test/")
for file in glob.glob("*.jpg"):
    print(file)
    detector.onImage(file)
    time.sleep(3)
from Detector import *
detector = Detector()

#%%

'''
Merge two json files as follows:
# python merge_prime.py result1.json result2.json train_dataset.json
merge_prime.py is stored in the IS folder.
'''

#%%

import time
import glob, os
# os.chdir("new_images/")
# for file in glob.glob("*.jpg"):
#     print(file)
#     detector.onImage(file)
#     time.sleep(3)
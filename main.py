from Detector import *
detector = Detector()

#%%

'''
Merge two json files as follows:
# python merge.py result1.json result2.json OUTPUT_JSON.json
merge.py is stored in the IS folder.
'''

'''
Correct the filepath for all images and save the new json file as 'train_dataset'
'''
import json
with open("OUTPUT_JSON.json") as f:
    dict = json.load(f)
    for i, j in enumerate(dict['images']):
        str_x = dict['images'][i]['file_name']
        dict['images'][i]['file_name'] = str_x.replace('images\\7/', '')

output_file = "train_dataset"
with open(output_file, 'w') as f:
        json.dump(dict, f)

#%%

import time
import glob, os
# os.chdir("new_images/")
# for file in glob.glob("*.jpg"):
#     print(file)
#     detector.onImage(file)
#     time.sleep(3)
""" Script demonstrating the Face2Text dataset. 
    Please activate The DeepLearning environment before executing this script.
"""

import json
import cv2
import os

descriptions_file = "Face2Text/face2text_v0.1/raw.json"

# load the json data:
with open(descriptions_file, "r") as j_desc:
    annos = json.load(j_desc)

# now iterate over the data and display it:
for annot in annos:
    temp = annot["image"][:-3]
    img_file = os.path.join("lfw", temp+"jpg")
    img = cv2.imread(img_file)
    if img is not None:
        descriptions = annot["descriptions"]
        # print all the descriptions:
        print("\nImage_Id:", annot["img_id"])
        for (i, description) in enumerate(descriptions, 1):
            print("%d.) %s" %(i, description["text"]))

        # show the image on screen
        cv2.imshow("Face:", img)
        cv2.waitKey(0)

print("Data display is over ... :)")
cv2.destroyAllWindows()

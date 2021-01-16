import csv
import os
import shutil

dest = "awe_data"


os.makedirs(dest, exist_ok=True)
with open('awe-translation.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    headers = True
    for full, img, sub in spamreader:
        if headers:
            headers = False
            continue

        type_of_data = "train"
        if full.startswith("test"):
            type_of_data = "val"

        subject, img_id = img.split('/')
        os.makedirs("{}/{}/{}".format(dest, type_of_data, subject), exist_ok=True)
        shutil.copy(
            "awe/{}".format(img), "{}/{}/{}".format(dest, type_of_data, subject)
        )

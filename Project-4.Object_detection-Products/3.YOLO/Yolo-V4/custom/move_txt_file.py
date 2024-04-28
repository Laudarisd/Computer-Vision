import shutil
import os
import glob
dst = "./labels"

try:
    os.makedirs('labels') 
except OSError:
       pass

for txt_file in glob.iglob('./annotations/*.txt'):
    shutil.move(txt_file, dst)

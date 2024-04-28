import os
import shutil
import glob
# your custom path
root = '3. final_img'


barcode_list = os.listdir(root)

for barcode in barcode_list:
    barcode_path = os.path.join(root, barcode)
    sub_list = glob.glob(barcode_path + '/*')
    if not '*.jpg' in sub_list:
        for sub in sub_list:
            img_list = os.listdir(sub)
            for img in img_list:
                img_path = os.path.join(sub, img)
                os.rename(img_path, '%s_%s' %(sub, img))
            shutil.rmtree(sub)
    print 'merge done', barcode

import random
import os
import sys


data_path = '/home/sudip/torch/Pytorch_tutorial/data/'
seed_imgs = 'images/'
seed_labels = 'Annotations'

seed_imgs_path = os.path.join(data_path, seed_imgs)


# test 데이터 나눌 필요 있을 때 사용
#try:
#    if not(os.path.isdir('./4k_data_raw/test')):
#        os.makedirs(os.path.join('./4k_data_raw/test'))
#except OSError as e:
#    if e.errno != errno.EEXIST:
#        raise

# path 수정 해야함
for file in ['train.txt', 'val.txt', 'trainval.txt']:
    if os.path.isfile(file):
        os.remove(file)


comb_list = sorted(os.listdir(seed_imgs_path))  # trainval
#label_list = sorted(os.listdir(label_path))
cnt = int(len(comb_list) * 0.1)

random_list = random.sample(comb_list, cnt)   # val
train_list = list(set(comb_list) - set(random_list))  # train



for train_file in train_list:
    filename, ext = train_file.split('.')
    with open ('train.txt', 'a') as f:
        f.write('{}\n'.format(str(filename)))

for val_file in random_list:
    filename, ext = val_file.split('.')
    with open ('val.txt', 'a') as f:
        f.write('{}\n'.format(str(filename)))

for trainval_file in comb_list:
    filename, ext = trainval_file.split('.')
    with open ('trainval.txt', 'a') as f:
        f.write('{}\n'.format(str(filename)))



import numpy as np
import os
from scipy import misc
import matplotlib.pyplot as plt
from scipy.misc import imread
import numpy, collections
import glob

os.chdir("D:/research/deeplab/datasets/anotacije_new")

tmp = np.zeros((56, 1))

image_list = glob.glob('*.png')
for file in image_list: #assuming gif
    a = imread(file)
    unique2, counts = numpy.unique(a, return_counts=True)
    tmp[unique2,0]= tmp[unique2,0] + counts
    print(file)


#tmp = tmp[tmp>0]
oldRange = max(tmp) - min(tmp)
newRange = 15 

tmp_new = (tmp * newRange / oldRange) 
utezi = 15-tmp_new

prazen_string = ''
for i in range(0,len(utezi),1):
    prazen_string = prazen_string + 'tf.to_float(tf.equal(scaled_labels,'+ str(i)+')) *' +str(utezi[i]) +'+'
	
python train.py \ --logtostderr \ --training_number_of_steps=300 \ --train_split="train" \ --model_variant="xception_65" \ --atrous_rates=6 \ --atrous_rates=12 \ --atrous_rates=18 \ --output_stride=16 \ --decoder_output_stride=4 \ --train_crop_size=513 \ --train_crop_size=513 \ --train_batch_size=2 \ --dataset="fashion" \ --tf_initial_checkpoint='/traindeeplab/deeplabv3_mnv2_cityscapes_train/model.ckpt' \ --train_logdir='/train_logdir' \ --dataset_dir='/TRF_record'
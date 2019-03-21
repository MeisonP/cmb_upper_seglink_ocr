# README

## tfrecords prepare: cmb_to_tfrecords.py
* datasets
> tfrecord:
> training_dataset:./image/201711070000000001001.jpg, ....
> training_ground_truth:
>	




## trianing command:
'''
$ python train_seglink.py --train_dir=./ckpt \
                        --learning_rate=0.00001 \
                        --gpu_memory_fraction=1 \
                        --train_image_width=512 \
                        -train_image_height=512 \
                        --batch_size=2 \
                        --dataset_dir=./datasets/tfrecord \
                        --dataset_name=cmb \
                        --dataset_split_name=train \
                        --train_with_ignored=0 \
                        --using_moving_average=0
'''

## test command:
'''
put the trained model into the path ./model/ at first,
$ python test_seglink.py --checkpoint_path=./model/model.ckpt-217867 \
                       --seg_conf_threshold=0.8 \
                       --link_conf_threshold=0.5 \
                       --dataset_dir=./image
'''


## visualization of test result:

'''
modify the det path at first,
$ python visualize_detection_result.py --image=./image  \
                      --det=./model/result_test/model.ckpt-217867/seg_link_conf_th_0.800000_0.500000/txt  \
                      --output=./output/
'''
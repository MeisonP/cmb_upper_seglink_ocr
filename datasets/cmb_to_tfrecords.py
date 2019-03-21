# encoding=utf-8
import numpy as np
import pandas as pd
import tensorflow as tf
import util
from dataset_utils import int64_feature, float_feature, bytes_feature, convert_to_example
        

def cvt_to_tfrecords(output_path , data_path, gt_path):
    image_names = util.io.ls(data_path, '.jpg')
    print "{0} images found in {1}".format(len(image_names), data_path)

    with tf.python_io.TFRecordWriter(output_path) as tfrecord_writer:
        for idx, image_name in enumerate(image_names):
            oriented_bboxes = []
            bboxes = []
            labels = []
            labels_text = []
            ignored = []
            path = util.io.join_path(data_path, image_name)
            print "\tconverting image: {0}/{1} {2}".format(idx, len(image_names), image_name)
            image_data = tf.gfile.FastGFile(path, 'r').read()
            
            image = util.img.imread(path, rgb=True)
            shape = image.shape
            h, w = shape[0:2]
            h *= 1.0
            w *= 1.0
            image_name = util.str.split(image_name, '.')[0] # without .jpg

            print image_name
            # read ground-truth

            gt_name = 'res_' + image_name + '.txt'
            gt_filepath = util.io.join_path(gt_path, gt_name)
            lines = util.io.read_lines(gt_filepath)

            for line in lines:
                line = util.str.remove_all(line, '\xef\xbb\xbf')
                gt = util.str.split(line, ',')
                oriented_box = [int(gt[i]) for i in range(8)]
                # oriented_box = np.asarray(oriented_box) / ([w, h] * 4)
                oriented_box = np.asarray(oriented_box) / ([w, h, w, h, w, h, w, h])
                oriented_bboxes.append(oriented_box)

                xs = oriented_box.reshape(4, 2)[:, 0]
                ys = oriented_box.reshape(4, 2)[:, 1]
                xmin = xs.min()
                xmax = xs.max()
                ymin = ys.min()
                ymax = ys.max()
                bboxes.append([xmin, ymin, xmax, ymax])
                ignored.append(util.str.contains(gt[-1], '###'))

                # might be wrong here, but it doesn't matter because the label is not going to be used in detection
                labels_text.append(gt[-1])
                labels.append(1)
            example = convert_to_example(image_data, image_name, labels, ignored, labels_text, bboxes, oriented_bboxes,
                                         shape)
            tfrecord_writer.write(example.SerializeToString())

            """
            df = pd.read_csv("./training_ground_truth/new_gt.csv")
            filename = image_name+".jpg"
            condition = df['name'] == filename
            subdf = df[condition]
            xmin, ymin, xmax, ymax = subdf.iloc[:, 1:5].values[0]

            oriented_bboxes.append([xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax])

            proportion_xmin, proportion_ymin, proportion_xmax, proportion_ymax = np.array([xmin, ymin, xmax, ymax]) / [w, h, w, h]
            bboxes.append([proportion_xmin, proportion_ymin, proportion_xmax, proportion_ymax])
            # bboxes should be a proportion of axis

            # ignored.append(util.str.contains(gt[-1], '###'))

            ignored.append(0)
            # might be wrong here, but it doesn't matter because the label is not going to be used in detection
            labels_text.append(1)
            labels.append(1)
            example = convert_to_example(image_data,
                                         image_name,
                                         labels,
                                         ignored,
                                         labels_text,
                                         bboxes,
                                         oriented_bboxes,
                                         shape)
            tfrecord_writer.write(example.SerializeToString())
            """


if __name__ == "__main__":
    root_dir = util.io.get_absolute_path('./')
    output_dir = util.io.get_absolute_path('./tfrecord/')
    util.io.mkdir(output_dir)

    training_data_dir = util.io.join_path(root_dir, 'training_dataset/image')
    training_gt_dir = util.io.join_path(root_dir, 'training_dataset/upper_test_batch1_qwertxt')
    cvt_to_tfrecords(output_path=util.io.join_path(output_dir, 'cmb_train.tfrecord'),
                     data_path=training_data_dir, gt_path=training_gt_dir)

#     test_data_dir = util.io.join_path(root_dir, 'ch4_test_images')
#     test_gt_dir = util.io.join_path(root_dir,'ch4_test_localization_transcription_gt')
#     cvt_to_tfrecords(output_path = util.io.join_path(output_dir, 'icdar2015_test.tfrecord'),
# data_path = test_data_dir, gt_path = test_gt_dir)

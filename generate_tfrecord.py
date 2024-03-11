"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=train.record
  # Create test data:
  python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=test.record
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow.compat.v1 as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('image_dir', '', 'Path to images')
FLAGS = flags.FLAGS


# TO-DO replace this with label map
def class_text_to_int(row_label):
    VOC_LABELS = {
        'none': (0, 'Background'),
        'Nấm rơm': (1, 'Nấm rơm'),
        'Nấm kim châm': (2, 'Nấm kim châm'),
        'Nấm hương': (3, 'Nấm hương'),
        'Nấm hầu thủ (nấm đầu khỉ': (4, 'Nấm hầu thủ (nấm đầu khỉ)'),
        'Nấm mỡ': (5, 'Nấm mỡ'),
        'Nấm thái dương': (6, 'Nấm thái dương'),
	    'Nấm linh chi': (7, 'Nấm linh chi'),
        'Nấm tràm': (8, 'Nấm tràm'),
	    'Nấm bào ngư': (9, 'Nấm bào ngư'),
	    'Nấm thông': (10, 'Nấm thông'),
	    'Nấm tuyết': (11, 'Nấm tuyết'),
	    'Nấm Deadly Dapperling': (12, 'Nấm Deadly Dapperling'),
	    'Nấm Tán Bay (Fly Agaric)': (13, 'Nấm Tán Bay (Fly Agaric)'),
	    'Nấm đôi cánh thiên thần': (14, 'Nấm đôi cánh thiên thần'),
	    'Nấm Conocybe Filaris': (15, 'Nấm Conocybe Filaris'),
	    'Nấm bàn tay tử thần': (16, 'Nấm bàn tay tử thần'),
	    'Nấm False Morel': (17, 'Nấm False Morel'),
	    'Nấm Morel': (18, 'Nấm Morel'),
	    'Nấm hoàng đế': (19, 'Nấm hoàng đế'),}
    return VOC_LABELS[row_label][0]
    # if row_label == 'mobile':
    #     return 1
    # else:
    #     None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    path = os.path.join(FLAGS.image_dir)
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()
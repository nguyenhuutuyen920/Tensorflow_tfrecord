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
        'cow-Brahmans': (1, 'cow-Brahmans'),
        'cow-lai-sua-vietnam': (2, 'cow-lai-sua-vietnam'),
        'cow-bay': (3, 'cow-bay'),
        'cow-thuy-dien': (4, 'cow-thuy-dien'),
        'cow-vang-vietnam': (5, 'cow-vang-vietnam'),
        'cow-sung': (6, 'cow-sung'),
	'cow-nepan': (7, 'cow-nepan'),
        'cow-vietnam': (8, 'cow-vietnam'),
	'cow-nauy': (9, 'cow-nauy'),
	'cow-sua-01': (10, 'cow-sua-01'),
	'cow-phap': (11, 'cow-phap'),
	'cow-yellow': (12, 'cow-yellow'),
	'cow-tot': (13, 'cow-tot'),
	'cow-black': (14, 'cow-black'),
	'cow-yellows': (8, 'cow-yellows'),
	'cow-power': (15, 'cow-power'),
	'cow-twocow': (16, 'cow-twocow'),
	'cow-foudes': (17, 'cow-foudes'),
	'cow-japan': (18, 'cow-japan'),
	'cow-ghi': (19, 'cow-ghi'),
	'cow-china': (20, 'cow-china'),
	'cow-faroar': (21, 'cow-faroar'),
	'cow-lao': (22, 'cow-lao'),
	'cow-foask': (23, 'cow-foask'),
	'cow-chile': (24, 'cow-chile'),
	'cow-vang': (25, 'cow-vang'),
	'cow-my': (26, 'cow-my'),
	'cow-korea': (27, 'cow-korea'),
	'cow-luc-dien': (28, 'cow-luc-dien'),
	'cow-duc': (29, 'cow-duc'),
	'cow-Jersey': (30, 'cow-Jersey'),
	'cow-Brahman': (31, 'cow-Brahman'),
	'cow-Sahiwal': (32, 'cow-Sahiwal'),
	'cow-Droughmaster': (33, 'cow-Droughmaster'),
	'cow-Angus': (34, 'cow-Angus'),
	'cow-Charolais': (35, 'cow-Charolais'),
	'cow-Limousin': (36, 'cow-Limousin'),
	'cow-bi': (37, 'cow-bi'),
	'cow-kobe': (38, 'cow-kobe'),
	'cow-ba-lan': (39, 'cow-ba-lan'),
	'cow-Blonde-Aquitaine': (40, 'cow-Blonde-Aquitaine'),
	'cow-sua-vietnam': (41, 'cow-sua-vietnam'),
	'cow-three-cow': (42, 'cow-three-cow'),
	'cow-sua-russia': (43, 'cow-sua-russia'),
	'cow-lai-sua': (44, 'cow-lai-sua'),
	'cow-Prasit': (44, 'cow-Prasit'),
	'cow-ha-lan': (45, 'cow-ha-lan'),}
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
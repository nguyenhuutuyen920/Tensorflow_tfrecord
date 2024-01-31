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
	'cow-yellows': (15, 'cow-yellows'),
	'cow-power': (16, 'cow-power'),
	'cow-twocow': (17, 'cow-twocow'),
	'cow-foudes': (18, 'cow-foudes'),
	'cow-japan': (19, 'cow-japan'),
	'cow-ghi': (20, 'cow-ghi'),
	'cow-china': (21, 'cow-china'),
	'cow-faroar': (22, 'cow-faroar'),
	'cow-lao': (23, 'cow-lao'),
	'cow-foask': (24, 'cow-foask'),
	'cow-chile': (25, 'cow-chile'),
	'cow-vang': (26, 'cow-vang'),
	'cow-my': (27, 'cow-my'),
	'cow-korea': (28, 'cow-korea'),
	'cow-luc-dien': (29, 'cow-luc-dien'),
	'cow-duc': (30, 'cow-duc'),
	'cow-Jersey': (31, 'cow-Jersey'),
	'cow-Brahman': (32, 'cow-Brahman'),
	'cow-Sahiwal': (33, 'cow-Sahiwal'),
	'cow-Droughmaster': (34, 'cow-Droughmaster'),
	'cow-Angus': (35, 'cow-Angus'),
	'cow-Charolais': (36, 'cow-Charolais'),
	'cow-Limousin': (37, 'cow-Limousin'),
	'cow-bi': (38, 'cow-bi'),
	'cow-kobe': (39, 'cow-kobe'),
	'cow-ba-lan': (40, 'cow-ba-lan'),
	'cow-Blonde-Aquitaine': (41, 'cow-Blonde-Aquitaine'),
	'cow-sua-vietnam': (42, 'cow-sua-vietnam'),
	'cow-three-cow': (43, 'cow-three-cow'),
	'cow-sua-russia': (44, 'cow-sua-russia'),
	'cow-lai-sua': (45, 'cow-lai-sua'),
	'cow-Prasit': (46, 'cow-Prasit'),
	'cow-ha-lan': (47, 'cow-ha-lan'),}
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
import tensorflow as tf
from utils import *


tf.enable_eager_execution()

if __name__ == '__main__':
    # Create TF Records
    train, val, test = get_db_sets('db')
    create_tf_record(train, 'train')
    create_tf_record(train, 'val')
    create_tf_record(train, 'test')

    # Create dataset
    dataset = create_dataset(path='./db/train.tfrecords',
                             buffer_size=64,
                             batch_size=64,
                             num_epochs=64)

    iterator = dataset.make_one_shot_iterator()

    # Read single image
    img = iterator.get_next()
    show_img(img[0])

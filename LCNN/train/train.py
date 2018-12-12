import tensorflow as tf
import numpy as np
import cv2


def concat_rgb_and_depth(root_folder, rgb_file_txt, depth_file_txt):
    rgb_imgs = []
    depth_imgs = []
    imgs = []
    labels = []
    with open(rgb_file_txt) as f:
        for line in f.readlines():
            rgb = cv2.imread(root_folder + line.split(' ')[0], 0)
            rgb_imgs.append(rgb)
            labels.append(line.split(' ')[1].strip('\n'))
    with open(depth_file_txt) as f:
        for line in f.readlines():
            depth = cv2.imread(root_folder + line.split(' ')[0], 0)
            depth_imgs.append(depth)
    for i in range(len(rgb_imgs)):
        print(i)
        merge_img = np.concatenate((rgb_imgs[i][:, :, None], depth_imgs[i][:, :, None]), axis=2)
        imgs.append(merge_img)
    return imgs, labels


def write_tfrecord(root_folder, rgb_file_txt, depth_file_txt, tfrecord_path):
    writer = tf.python_io.TFRecordWriter(tfrecord_path)
    imgs, labels = concat_rgb_and_depth(root_folder, rgb_file_txt, depth_file_txt)
    for image, label in zip(imgs, labels):
        label = int(label)
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tostring()])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
            }
        ))
        writer.write(record=example.SerializeToString())
    writer.close()


def read_tfrecord(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features={
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
    })
    # byteslist类型的需要重新解码，int64list类型则不用
    image = tf.decode_raw(features['image'], tf.float32)
    label = features['label']
    # image.set_shape([128*128*2])
    image = tf.reshape(image, [128, 128, 2])
    image = image / 255.0
    return image, label


def input(batch_size, tfrecord_path):
    filename_queue = tf.train.string_input_producer([tfrecord_path])
    image, label = read_tfrecord(filename_queue)
    images, labels = tf.train.shuffle_batch([image, label], batch_size, min_after_dequeue=1,
                                            capacity=1 + 3 * batch_size, num_threads=1)
    return images, labels


def main():
    # rgb_file_txt = '/Volumes/Untitled/eaststation/test/test_3Dtexture.txt'
    # depth_file_txt = '/Volumes/Untitled/eaststation/test/test_3Ddepth.txt'
    # root_folder = '/Volumes/Untitled/eaststation/test/'
    rgb_file_txt = '../../eaststation/train_3Dtexture.txt'
    depth_file_txt = '../../eaststation/train_3Ddepth.txt'
    root_folder = '../../eaststation/'
    tfrecord_path = '../../eaststation/eaststation.tfrecord'
    write_tfrecord(root_folder, rgb_file_txt, depth_file_txt, tfrecord_path)
    images, label = input(1, tfrecord_path)
    step = 0
    with tf.Session() as sess:
        init_op = tf.group(tf.local_variables_initializer(),tf.global_variables_initializer())
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        while step < 3:
            imgs = sess.run(images)
            print(imgs)
            step +=1
        coord.request_stop()
        coord.join(threads)


if __name__ == "__main__":
    main()

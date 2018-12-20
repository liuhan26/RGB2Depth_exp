import tensorflow as tf
import numpy as np
import cv2
import os
from LCNN29 import LCNN29, LCNN9
import model as M
import argparse
import shutil
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='TensorFlow LightCNN29 training')
parser.add_argument('--lr', default='0.01', type=float, help='learning rate')
parser.add_argument('--epoch', default='80', type=int, help='training epochs')
parser.add_argument('--batch_size', default='128', type=int, help='min batch size')
parser.add_argument('--samples_num', default='33433', type=int, help='the number of total training samples')


def fill_hole(depth_im):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    depth_im_de = cv2.morphologyEx(depth_im, cv2.MORPH_OPEN, kernel)
    mask = np.less(depth_im, 1)
    depth_im_de = np.where(mask, depth_im, depth_im_de)
    return depth_im_de


def concat_rgb_and_depth(root_folder, rgb_file_txt, depth_file_txt):
    imgs = []
    labels = []
    f1 = open(rgb_file_txt)
    rgb_lines = f1.readlines()
    f2 = open(depth_file_txt)
    depth_lines = f2.readlines()
    i = 0
    for rgb_line, depth_line in zip(rgb_lines, depth_lines):
        rgb_path = root_folder + rgb_line.split('\t')[0]
        depth_path = root_folder + depth_line.split('\t')[0]
        label = rgb_line.split('\t')[1].strip('\n')
        if os.path.exists(rgb_path) and os.path.exists(depth_path):
            rgb = cv2.imread(rgb_path, 0)
            depth = cv2.imread(depth_path, 0)
            if rgb is not None and depth is not None:
                labels.append(label)
                # print(depth_path)
                i += 1
                print(i)
                depth_de = fill_hole(depth)
                merge_img = np.concatenate((rgb[:, :, None], depth_de[:, :, None]), axis=2)
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
    image = tf.decode_raw(features['image'], tf.uint8)  # 还原Image原本的格式，而不是直接转化为tf.float32
    label = features['label']
    image.set_shape([128 * 128 * 2])
    image = tf.reshape(image, [128, 128, 2])
    image = tf.cast(image, tf.float32) // 255
    return image, label


def input(batch_size, epochs, tfrecord_path):
    filename_queue = tf.train.string_input_producer([tfrecord_path], num_epochs=epochs)
    image, label = read_tfrecord(filename_queue)
    images, labels = tf.train.shuffle_batch([image, label], batch_size, min_after_dequeue=1000,
                                            capacity=1000 + 3 * batch_size, num_threads=2)
    return images, labels


def main():
    args = parser.parse_args()
    # rgb_file_txt = '/Volumes/Untitled/eaststation/test/test_3Dtexture.txt'
    # depth_file_txt = '/Volumes/Untitled/eaststation/test/test_3Ddepth.txt'
    # root_folder = '/Volumes/Untitled/eaststation/test/'
    rgb_file_txt = '/home/wtx/RGBD_dataset/eaststation/train/train_3Dtexture.txt'
    depth_file_txt = '/home/wtx/RGBD_dataset/eaststation/train/train_3Ddepth.txt'
    root_folder = '/home/wtx/RGBD_dataset/eaststation/train/crop_image_realsense_128_128/'
    tfrecord_path = '../../eaststation/eaststation.tfrecord'

    if not os.path.exists(tfrecord_path):
        write_tfrecord(root_folder, rgb_file_txt, depth_file_txt, tfrecord_path)
    images, labels = input(args.batch_size, args.batch_size, tfrecord_path)
    loss, acc = LCNN9(images, labels)
    l2_loss = tf.losses.get_regularization_loss()
    loss += l2_loss
    # global_step = tf.Variable(0, trainable=False)
    # learning_rate = tf.train.exponential_decay(learning_rate=args.lr, global_step=global_step,
    #                                           decay_steps=10*args.samples_num/args.batch_size,
    #                                           decay_rate=0.46)
    train_op = tf.train.MomentumOptimizer(0.0001, 0.9).minimize(loss)
    # train_op = tf.train.AdamOptimizer(0.0001).minimize(loss)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess, epoch, step = M.loadSess('../tfmodel/', sess)
        saver = tf.train.Saver()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            while not coord.should_stop():
                _, loss_value = sess.run([train_op, loss])
                step += 1
                if (args.batch_size * step) % args.samples_num == 0:
                    epoch += 1
                print('epoch = %d  iter = %d loss = %.2f' % (epoch, step, loss_value))
                print('accuracy = %.2f' % acc)
                if step % 100 == 0:
                    save_path = '../tfmodel/Epoc_' + str(epoch) + '_' + 'Iter_' + str(step) + '.cpkt'
                    saver.save(sess, save_path)
                    save_path2 = save_path + '.meta'
                    save_path3 = save_path + '.index'
                    save_path4 = save_path + '.data-00000-of-00001'
                    save_path5 = '../tfmodel/checkpoint'

                    shutil.copy(save_path2, save_path2.replace('../tfmodel/', '../backup/'))
                    shutil.copy(save_path3, save_path3.replace('../tfmodel/', '../backup/'))
                    shutil.copy(save_path4, save_path4.replace('../tfmodel/', '../backup/'))
                    shutil.copy(save_path5, save_path5.replace('../tfmodel/', '../backup/'))

        except tf.errors.OutOfRangeError:
            print('Done training for %d steps' % (step))
        finally:
            coord.request_stop()
        coord.join(threads)


if __name__ == "__main__":
    main()

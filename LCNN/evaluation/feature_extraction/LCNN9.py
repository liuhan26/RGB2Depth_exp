import model as M
import tensorflow as tf


def LCNN9():
    img_holder = tf.placeholder(tf.float32, [None, 128, 128, 1])
    mod = M.Model(img_holder, [None, 128, 128, 1])

    mod.conv_layer(5, 96, activation=1)
    mod.maxpooling_layer(2, 2)

    mod.conv_layer(1, 96, activation=1)
    mod.conv_layer(3, 192, activation=1)
    mod.maxpooling_layer(2, 2)

    mod.conv_layer(1, 192, activation=1)
    mod.conv_layer(3, 384, activation=1)
    mod.maxpooling_layer(2, 2)

    mod.conv_layer(1, 384, activation=1)
    mod.conv_layer(3, 256, activation=1)
    mod.maxpooling_layer(2, 2)

    mod.conv_layer(1, 256, activation=1)
    mod.conv_layer(3, 256, activation=1)
    mod.maxpooling_layer(2, 2)

    mod.flatten()
    mod.fcnn_layer(512)
    # mod.dropout(1)
    # mod.fcnn_layer(2)
    feature_layer = mod.get_current_layer()[0]
    # acc = mod.accuracy(lab_holder)

    return feature_layer, img_holder


with tf.variable_scope('LCNN9'):
    feature_layer, img_holder = LCNN9()


sess = tf.Session()
model_path = '../tfmodel/Epoc_194_Iter_102000.cpkt'
M.loadSess(model_path, sess)


def eval(imgs):
    feature = sess.run(feature_layer, feed_dict={img_holder: imgs})
    return feature


def __exit__():
    sess.close()
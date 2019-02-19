import model as M


def _LCNN9(img_holder, lab_holder):
    mod = M.Model(img_holder, [None, 128, 128, 2])

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
    mod.dropout(1)
    mod.fcnn_layer(2)

    acc = mod.accuracy(lab_holder)

    return acc
""" 
ChronoRoot: High-throughput phenotyping by deep learning reveals novel temporal parameters of plant root system architecture
Copyright (C) 2020 Nicolás Gaggion

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import tensorflow as tf


def repeat(tensor, repeats):
    """
    Args:

    input: A Tensor. 1-D or higher.
    repeats: A list. Number of repeat for each dimension, length must be the same as the number of dimensions in input

    Returns:

    A Tensor. Has the same type as input. Has the shape of tensor.shape * repeats

    Code extracted from: https://github.com/tensorflow/tensorflow/issues/8246
    """
    with tf.compat.v1.variable_scope("repeat"):
        expanded_tensor = tf.expand_dims(tensor, -1)
        multiples = [1] + repeats
        tiled_tensor = tf.tile(expanded_tensor, multiples=multiples)
        repeated_tensor = tf.reshape(tiled_tensor, [a * b for a, b in zip(tensor.get_shape(), repeats)])

    return repeated_tensor


def upsample2d(tensor, factor):
    """
      Receives a 4D tensor (batch_num, w, h, channels) and upsamples it by repeating every element "factor" times in the "w" and "h" axes.
    """
    shape = tf.shape(tensor)
    height = shape[1]
    width = shape[2]
    new_height = height * factor
    new_width = width * factor
    return tf.image.resize_nearest_neighbor(tensor, [new_height, new_width])


def dropout(tensor, rate, isTrain, name):
    with tf.compat.v1.variable_scope(name):
        return tf.layers.dropout(tensor, rate=rate, training=isTrain, name=name)


def conv2d(x, name, dim, k, s, p, bn, af, is_train):
    """
      2D Convolution in tensorflow

    :param x: input tensor
    :param name: name of the layer
    :param dim: number of filters of the output
    :param k: kernel size
    :param s: stride
    :param p: padding strategy ("SAME", "VALID")
    :param bn: use batch normalization (boolean)
    :param af: activation functione. None if no activation function.
    :param is_train: indicates if we are at training time or not (better to use a boolean placeholder).

    :return: the convolutional layer
    """
    with tf.compat.v1.variable_scope(name):
        w = tf.compat.v1.get_variable('weight', [k, k, x.get_shape()[-1], dim],
                                      initializer=tf.truncated_normal_initializer(stddev=0.01))
        x = tf.nn.conv2d(x, w, [1, s, s, 1], p)

        if bn:
            x = batch_norm(x, "bn", is_train=is_train)
        else:
            b = tf.compat.v1.get_variable('biases', [dim],
                                          initializer=tf.constant_initializer(0.))
            x += b

        if af:
            x = af(x)

    return x


def batch_norm(x, name, momentum=0.9, epsilon=1e-5, is_train=True):
    return tf.contrib.layers.batch_norm(x,
                                        decay=momentum,
                                        updates_collections=None,
                                        epsilon=epsilon,
                                        scale=True,
                                        is_training=is_train,
                                        scope=name)


def pixel_wise_softmax(output_map):
    with tf.name_scope("pixel_wise_softmax"):
        max_axis = tf.reduce_max(output_map, axis=3, keepdims=True)
        exponential_map = tf.exp(output_map - max_axis)
        normalize = tf.reduce_sum(exponential_map, axis=3, keepdims=True)
        return exponential_map / normalize


## Loss Functions

def dice_coe_c1(output, target, loss_type='sorensen', axis=(1, 2), smooth=1e-5):
    """Soft dice (Sørensen or Jaccard) coefficient for comparing the similarity
    of two batch of data, usually be used for binary image segmentation
    i.e. labels are binary. The coefficient between 0 to 1, 1 means totally match.

    Parameters
    -----------
    output : Tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    target : Tensor
        The target distribution, format the same with `output`.
    loss_type : str
        ``jaccard`` or ``sorensen``, default is ``jaccard``.
    axis : tuple of int
        All dimensions are reduced, default ``[1,2,3]``.
    smooth : float
        This small value will be added to the numerator and denominator.
            - If both output and target are empty, it makes sure dice is 1.
            - If either output or target are empty (all pixels are background), 
            dice = ```smooth/(small_value + smooth)``, then if smooth is very small, 
            dice close to 0 (even the image values lower than the threshold),
            so in this case, higher smooth can have a higher dice.

    Examples
    ---------
    >>> outputs = tl.act.pixel_wise_softmax(network.outputs)
    >>> dice_loss = 1 - tl.cost.dice_coe(outputs, y_)

    """
    output = output[:, :, :, 1]
    target = target[:, :, :, 1]
    inse = tf.reduce_sum(output * target, axis=axis)
    if loss_type == 'jaccard':
        l = tf.reduce_sum(output * output, axis=axis)
        r = tf.reduce_sum(target * target, axis=axis)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(output, axis=axis)
        r = tf.reduce_sum(target, axis=axis)
    else:
        raise Exception("Unknow loss_type")

    dice = (2. * inse + smooth) / (l + r + smooth)
    ##
    dice = tf.reduce_mean(dice, name='dice_coe')
    return dice


def dice_hard_coe(output, target, threshold=0.5, axis=(1, 2, 3), smooth=1e-5):
    """Non-differentiable Sørensen–Dice coefficient for comparing the similarity
    of two batch of data, usually be used for binary image segmentation i.e. labels are binary.
    The coefficient between 0 to 1, 1 if totally match.

    Parameters
    -----------
    output : tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    target : tensor
        The target distribution, format the same with `output`.
    threshold : float
        The threshold value to be true.
    axis : tuple of integer
        All dimensions are reduced, default ``(1,2,3)``.
    smooth : float
        This small value will be added to the numerator and denominator, see ``dice_coe``.

    References
    -----------
    - `Wiki-Dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`__

    """
    output = tf.cast(output > threshold, dtype=tf.float32)
    target = tf.cast(target > threshold, dtype=tf.float32)
    inse = tf.reduce_sum(tf.multiply(output, target), axis=axis)
    l = tf.reduce_sum(output, axis=axis)
    r = tf.reduce_sum(target, axis=axis)
    hard_dice = (2. * inse + smooth) / (l + r + smooth)
    hard_dice = tf.reduce_mean(hard_dice, name='hard_dice')
    return hard_dice


def dice_hard_coe_c1(output, target, threshold=0.5, axis=(1, 2), smooth=1e-5):
    output = output[:, :, :, 1]
    target = target[:, :, :, 1]
    output = tf.cast(output > threshold, dtype=tf.float32)
    target = tf.cast(target > threshold, dtype=tf.float32)
    inse = tf.reduce_sum(tf.multiply(output, target), axis=axis)
    l = tf.reduce_sum(output, axis=axis)
    r = tf.reduce_sum(target, axis=axis)
    hard_dice = (2. * inse + smooth) / (l + r + smooth)
    hard_dice = tf.reduce_mean(hard_dice, name='hard_dice')
    return hard_dice


def dice_coe(output, target, loss_type='sorensen', axis=(1, 2), smooth=1e-5):
    inse = tf.reduce_sum(output * target, axis=axis)
    if loss_type == 'jaccard':
        l = tf.reduce_sum(output * output, axis=axis)
        r = tf.reduce_sum(target * target, axis=axis)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(output, axis=axis)
        r = tf.reduce_sum(target, axis=axis)
    else:
        raise Exception("Unknow loss_type")

    dice = (2. * inse + smooth) / (l + r + smooth)
    ##
    dice = tf.reduce_mean(dice, name='dice_coe')
    return dice


##Summary Functions

def addScalarValueToSummary(name, value, summaryWriter, step):
    """
      Writes down to a summary a scalar value
    """
    summary = tf.compat.v1.Summary(value=[
        tf.compat.v1.Summary.Value(tag=name, simple_value=value),
    ])
    summaryWriter.add_summary(summary, step)


## Extra functions


def hard_indicator(inputs):
    return 1e10 * tf.cast(inputs > 0.5, tf.float32)


def resUnit(x, i_name, n_kernels, ksize, stride, padding, activation_function, isTrain):
    bn1 = batch_norm(x, "bn%s_a" % i_name, is_train=isTrain)
    af1 = activation_function(bn1)
    conv1 = conv2d(af1, "conv%s_a" % i_name, n_kernels, ksize, stride, "SAME",
                   bn=None, af=None, is_train=isTrain)

    bn2 = batch_norm(conv1, "bn%s_b" % i_name, is_train=isTrain)
    af2 = activation_function(bn2)
    conv2 = conv2d(af2, "conv%s_b" % i_name, n_kernels, ksize, 1, "SAME",
                   bn=None, af=None, is_train=isTrain)

    x_ = conv2d(x, "conv%s_x" % i_name, n_kernels, 1, stride, "SAME",
                bn=False, af=None, is_train=isTrain)

    return tf.add(x_, conv2)


import tensorflow as tf


class SoftSkeletonizeTF:
    def __init__(self, num_iter=40):
        self.num_iter = num_iter

    def soft_erode(self, img):
        """
        Érosion douce utilisant le MaxPool négatif.
        img shape: [Batch, H, W, C]
        Traduction exacte de la logique PyTorch (Cross-shaped erosion)
        """
        # p1 : Kernel (3, 1) -> érosion verticale
        # ksize=[1, 3, 1, 1] signifie [Batch, H, W, Channel]
        p1 = -tf.nn.max_pool(-img, ksize=[1, 3, 1, 1], strides=[1, 1, 1, 1], padding='SAME')

        # p2 : Kernel (1, 3) -> érosion horizontale
        p2 = -tf.nn.max_pool(-img, ksize=[1, 1, 3, 1], strides=[1, 1, 1, 1], padding='SAME')

        # On prend le min des deux (comme dans votre code torch.min(p1, p2))
        return tf.minimum(p1, p2)

    def soft_dilate(self, img):
        """
        Dilatation douce.
        img shape: [Batch, H, W, C]
        Kernel (3, 3)
        """
        return tf.nn.max_pool(img, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')

    def soft_open(self, img):
        return self.soft_dilate(self.soft_erode(img))

    def soft_skel(self, img):
        img1 = self.soft_open(img)
        skel = tf.nn.relu(img - img1)

        # Déroulement de la boucle dans le graphe TF
        for j in range(self.num_iter):
            img = self.soft_erode(img)
            img1 = self.soft_open(img)
            delta = tf.nn.relu(img - img1)
            # skel = skel + relu(delta - skel * delta)
            skel = skel + tf.nn.relu(delta - (skel * delta))

        return skel

    def __call__(self, img):
        return self.soft_skel(img)


def soft_dice_tf(y_true, y_pred, smooth=1.0):
    """
    Calcul du Dice Soft (similaire à votre fonction soft_dice).
    Attend des entrées [Batch, H, W, 1] (Foreground uniquement)
    """
    intersection = tf.reduce_sum(y_true * y_pred)
    coeff = (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)
    return 1. - coeff


def soft_cldice_loss(y_true, y_pred, iter_=10, alpha=0.5, smooth=1.0):
    """
    La loss combinée finale : Alpha * clDice + (1-Alpha) * Dice
    """
    # Instanciation du squelettiseur
    # Note: iter_ doit être ajusté selon l'épaisseur des racines (10 est standard)
    soft_skel_runner = SoftSkeletonizeTF(num_iter=iter_)

    # Calcul du Dice classique
    dice_loss = soft_dice_tf(y_true, y_pred, smooth)

    # Calcul des squelettes
    skel_pred = soft_skel_runner(y_pred)
    skel_true = soft_skel_runner(y_true)

    # Calcul T-Prec (Précision Topologique)
    # Somme sur tous les axes sauf le batch (ou global, selon votre implémentation PyTorch)
    # Ici on fait une somme globale pour la stabilité, comme dans votre code PyTorch original
    tprec = (tf.reduce_sum(skel_pred * y_true) + smooth) / (tf.reduce_sum(skel_pred) + smooth)

    # Calcul T-Sens (Sensibilité Topologique)
    tsens = (tf.reduce_sum(skel_true * y_pred) + smooth) / (tf.reduce_sum(skel_true) + smooth)

    # clDice
    cl_dice_val = 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens)

    return (1.0 - alpha) * dice_loss + alpha * cl_dice_val

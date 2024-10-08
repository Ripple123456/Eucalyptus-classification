
# import argparse
import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorpack import *
from tensorpack.tfutils import gradproc, optimizer, summary

from tensorpack.dataflow import DataFlow
from tensorpack import logger

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


train_dir = '/content/drive/My Drive/data/train'
validation_dir = '/content/drive/My Drive/data/validation'


IMAGE_SIZE = 224
WARP_TARGET_SIZE = 224
HALF_DIFF = (IMAGE_SIZE - WARP_TARGET_SIZE) // 2


class DirectoryIteratorFlow(DataFlow):
    def __init__(self, iterator, length):
        self.iterator = iterator
        self.length = length

    def __iter__(self):
        for batch in self.iterator:
            images, labels = batch
            yield images, labels

    def __len__(self):
        return self.length


def adjust_channels(image, num_channels):
    if num_channels == 1:
        # 转换为灰度图
        image = tf.image.rgb_to_grayscale(image)
    elif num_channels == 2:
        # 只保留前两个通道
        image = image[:, :, :2]
    elif num_channels == 3:
        # 已经是RGB图，不需要改变
        pass
    return image



def sample(img, coords):
    """
    """
    shape = img.get_shape().as_list()[1:]   # h, w, c
    batch = tf.shape(img)[0]
    shape2 = coords.get_shape().as_list()[1:3]  # h2, w2
    assert None not in shape2, coords.get_shape()
    max_coor = tf.constant([shape[0] - 1, shape[1] - 1], dtype=tf.float32)

    coords = tf.clip_by_value(coords, 0., max_coor)  # borderMode==repeat
    coords = tf.cast(coords, tf.int32)

    batch_index = tf.range(batch, dtype=tf.int32)
    batch_index = tf.reshape(batch_index, [-1, 1, 1, 1])
    batch_index = tf.tile(batch_index, [1, shape2[0], shape2[1], 1])    # bxh2xw2x1
    indices = tf.concat([batch_index, coords], axis=3)  # bxh2xw2x3
    sampled = tf.gather_nd(img, indices)
    return sampled


@layer_register(log_shape=True)
def GridSample(inputs, borderMode='repeat'):
    """
    
        tf.Tensor: a tensor named ``output`` of shape (N, H', W', C).
    """
    image, mapping = inputs
    assert image.get_shape().ndims == 4 and mapping.get_shape().ndims == 4
    input_shape = image.get_shape().as_list()[1:]
    assert None not in input_shape, \
        "Images in GridSample layer must have fully-defined shape"
    assert borderMode in ['repeat', 'constant']

    orig_mapping = mapping
    mapping = tf.maximum(mapping, 0.0)
    lcoor = tf.floor(mapping)
    ucoor = lcoor + 1

    diff = mapping - lcoor
    neg_diff = 1.0 - diff  # bxh2xw2x2

    lcoory, lcoorx = tf.split(lcoor, 2, 3)
    ucoory, ucoorx = tf.split(ucoor, 2, 3)

    lyux = tf.concat([lcoory, ucoorx], 3)
    uylx = tf.concat([ucoory, lcoorx], 3)

    diffy, diffx = tf.split(diff, 2, 3)
    neg_diffy, neg_diffx = tf.split(neg_diff, 2, 3)

    ret = tf.add_n([sample(image, lcoor) * neg_diffx * neg_diffy,
                    sample(image, ucoor) * diffx * diffy,
                    sample(image, lyux) * neg_diffy * diffx,
                    sample(image, uylx) * diffy * neg_diffx], name='sampled')
    if borderMode == 'constant':
        max_coor = tf.constant([input_shape[0] - 1, input_shape[1] - 1], dtype=tf.float32)
        mask = tf.greater_equal(orig_mapping, 0.0)
        mask2 = tf.less_equal(orig_mapping, max_coor)
        mask = tf.logical_and(mask, mask2)  # bxh2xw2x2
        mask = tf.reduce_all(mask, [3])  # bxh2xw2 boolean
        mask = tf.expand_dims(mask, 3)
        ret = ret * tf.cast(mask, tf.float32)
    return tf.identity(ret, name='output')


class Model(ModelDesc):
    def inputs(self):
        return [tf.TensorSpec((None, IMAGE_SIZE, IMAGE_SIZE, 2), tf.float32, 'input'),
                tf.TensorSpec((None,), tf.int32, 'label')]

    def build_graph(self, image, label):
        xys = np.array([(y, x, 1) for y in range(WARP_TARGET_SIZE)
                        for x in range(WARP_TARGET_SIZE)], dtype='float32')
        xys = tf.constant(xys, dtype=tf.float32, name='xys')    # p x 3

        image = image / 255.0 - 0.5  # bhw2

        def get_stn(image):
            stn = (LinearWrap(image)
                   .AvgPooling('downsample', 2)
                   .Conv2D('conv0', 20, 5, padding='VALID')
                   .MaxPooling('pool0', 2)
                   .Conv2D('conv1', 20, 5, padding='VALID')
                   .FullyConnected('fc1', 32)
                   .FullyConnected('fct', 6, activation=tf.identity,
                                   kernel_initializer=tf.constant_initializer(),
                                   bias_initializer=tf.constant_initializer([1, 0, HALF_DIFF, 0, 1, HALF_DIFF]))())
            # output 6 parameters for affine transformation
            stn = tf.reshape(stn, [-1, 2, 3], name='affine')  # bx2x3
            stn = tf.reshape(tf.transpose(stn, [2, 0, 1]), [3, -1])  # 3 x (bx2)
            coor = tf.reshape(tf.matmul(xys, stn),
                              [WARP_TARGET_SIZE, WARP_TARGET_SIZE, -1, 2])
            coor = tf.transpose(coor, [2, 0, 1, 3], 'sampled_coords')  # b h w 2
            sampled = GridSample('warp', [image, coor], borderMode='constant')
            return sampled

        with argscope([Conv2D, FullyConnected], activation=tf.nn.relu):
            with tf.variable_scope('STN1'):
                sampled1 = get_stn(image)
            with tf.variable_scope('STN2'):
                sampled2 = get_stn(image)

        # For visualization in tensorboard
        with tf.name_scope('visualization'):
            padded1 = tf.pad(sampled1, [[0, 0], [HALF_DIFF, HALF_DIFF], [HALF_DIFF, HALF_DIFF], [0, 0]])
            padded2 = tf.pad(sampled2, [[0, 0], [HALF_DIFF, HALF_DIFF], [HALF_DIFF, HALF_DIFF], [0, 0]])
            img_orig = tf.concat([image[:, :, :, 0], image[:, :, :, 1]], 1)  # b x 2h  x w
            transform1 = tf.concat([padded1[:, :, :, 0], padded1[:, :, :, 1]], 1)
            transform2 = tf.concat([padded2[:, :, :, 0], padded2[:, :, :, 1]], 1)
            stacked = tf.concat([img_orig, transform1, transform2], 2, 'viz')
            tf.summary.image('visualize',
                             tf.expand_dims(stacked, -1), max_outputs=30)

        sampled = tf.concat([sampled1, sampled2], 3, 'sampled_concat')
        logits = (LinearWrap(sampled)
                  .FullyConnected('fc1', 256, activation=tf.nn.relu)
                  .FullyConnected('fc2', 128, activation=tf.nn.relu)
                  .FullyConnected('fct', 19, activation=tf.identity)())
        tf.nn.softmax(logits, name='prob')

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        wrong = tf.cast(tf.logical_not(tf.nn.in_top_k(logits, label, 1)), tf.float32, name='incorrect_vector')
        summary.add_moving_summary(tf.reduce_mean(wrong, name='train_error'))

        wd_cost = tf.multiply(1e-5, regularize_cost('fc.*/W', tf.nn.l2_loss),
                              name='regularize_loss')
        summary.add_moving_summary(cost, wd_cost)
        return tf.add_n([wd_cost, cost], name='cost')

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=5e-4, trainable=False)
        opt = tf.train.AdamOptimizer(lr, epsilon=1e-3)
        return optimizer.apply_grad_processors(
            opt, [
                gradproc.ScaleGradient(('STN.*', 0.1)),
                gradproc.SummaryGradient()])


# def image_data_generator_with_channel_adjustment(generator, num_channels):
#     for batch_x, batch_y in generator:
#         batch_x_adjusted = adjust_channels(batch_x, num_channels)
#         yield batch_x_adjusted, batch_y


def image_data_generator_with_channel_adjustment(generator, num_channels):
    for batch_x, batch_y in generator:
        batch_x_adjusted = np.array([adjust_channels(x, num_channels) for x in batch_x])
        yield batch_x_adjusted, batch_y




def get_data(isTrain, batch_size, num_channels):
    if isTrain:
        datagen = ImageDataGenerator(rescale=1./255)
        data_dir = train_dir
    else:
        datagen = ImageDataGenerator(rescale=1./255)
        data_dir = validation_dir

    generator = datagen.flow_from_directory(
        data_dir,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=batch_size,
        class_mode='categorical'
    )
    adjusted_generator = image_data_generator_with_channel_adjustment(generator, num_channels)
    return DirectoryIteratorFlow(adjusted_generator, generator.samples // batch_size)



def view_warp(modelpath):
    pred = OfflinePredictor(PredictConfig(
        session_init=SmartInit(modelpath),
        model=Model(),
        input_names=['input'],
        output_names=['visualization/viz', 'STN1/affine', 'STN2/affine']))

    xys = np.array([[0, 0, 1],
                    [WARP_TARGET_SIZE, 0, 1],
                    [WARP_TARGET_SIZE, WARP_TARGET_SIZE, 1],
                    [0, WARP_TARGET_SIZE, 1]], dtype='float32')

    def draw_rect(img, affine, c, offset=(0, 0)):
        a = np.transpose(affine)  # 3x2
        a = (np.matmul(xys, a) + list(offset)).astype('int32')
        cv2.line(img, tuple(a[0][::-1]), tuple(a[1][::-1]), c)
        cv2.line(img, tuple(a[1][::-1]), tuple(a[2][::-1]), c)
        cv2.line(img, tuple(a[2][::-1]), tuple(a[3][::-1]), c)
        cv2.line(img, tuple(a[3][::-1]), tuple(a[0][::-1]), c)

    # ds = get_data(False)
    ds = get_data(False, num_channels=2)  # 添加 num_channels 参数
    ds.reset_state()
    for k in ds:
        img, label = k
        outputs, affine1, affine2 = pred(img)
        for idx, viz in enumerate(outputs):
            viz = cv2.cvtColor(viz, cv2.COLOR_GRAY2BGR)
            # Here we assume the second branch focuses on the first digit
            draw_rect(viz, affine2[idx], (0, 0, 255))
            draw_rect(viz, affine1[idx], (0, 0, 255), offset=[IMAGE_SIZE, 0])
            cv2.imwrite('{:03d}.png'.format(idx), (viz + 0.5) * 255)
        break








def get_config():
    log_dir = '/content/drive/My Drive/training_logs'
    logger.set_logger_dir(log_dir, action='k')

    dataset_train = get_data(True, batch_size=128, num_channels=2)
    dataset_test = get_data(False, batch_size=128, num_channels=2)

    return TrainConfig(
        model=Model(),
        data=QueueInput(dataset_train),
        callbacks=[
            ModelSaver(),
            InferenceRunner(dataset_test, [ScalarStats('cost'), ClassificationError()]),
            ScheduledHyperParamSetter('learning_rate', [(200, 1e-4)])
        ],
        steps_per_epoch=len(dataset_train),
        max_epoch=500,
    )






# 直接设置参数
args_gpu = ''  # 设置 GPU 编号，如果不使用 GPU 则留空
args_load = None  # 设置模型加载路径，如果没有预训练模型则为 None
args_view = False  # 设置是否查看模型，通常为 False

if args_gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = args_gpu

# 根据 args_view 的值执行不同的操作
if args_view:
    view_warp(args_load)
else:
    tf.reset_default_graph()
    config = get_config()
    if args_load:
        config.session_init = SmartInit(args_load)
    launch_train_with_config(config, SimpleTrainer())


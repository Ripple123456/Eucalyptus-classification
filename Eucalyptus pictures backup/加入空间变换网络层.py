#加入空间变换网络层

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add, Layer, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# 定义空间变换网络层
class SpatialTransformer(Layer):
    def __init__(self):
        super(SpatialTransformer, self).__init__()

        # 定位网络
        self.localization = tf.keras.Sequential([
            Conv2D(8, kernel_size=7, activation='relu'),
            MaxPooling2D(pool_size=2),
            Flatten(),
            Dense(32, activation='relu'),
            Dense(6, activation='linear', use_bias=False,
                  kernel_initializer=tf.keras.initializers.Zeros(),
                  bias_initializer=tf.keras.initializers.Constant([1, 0, 0, 0, 1, 0]))
        ])

    def call(self, x):
        # 应用定位网络
        theta = self.localization(x)
        theta = tf.reshape(theta, (-1, 2, 3))

        # 生成网格
        output_size = (x.shape[1], x.shape[2])
        grid = self._generate_grid(output_size, theta)

        # 采样
        x_transformed = self._grid_sample(x, grid)
        return x_transformed

    def _generate_grid(self, output_size, theta):
        # 生成变换后的网格
        height, width = output_size
        num_batch = tf.shape(theta)[0]

        # 生成网格坐标
        x = tf.linspace(-1.0, 1.0, width)
        y = tf.linspace(-1.0, 1.0, height)
        x_t, y_t = tf.meshgrid(x, y)

        # 扁平化
        x_t_flat = tf.reshape(x_t, [-1])
        y_t_flat = tf.reshape(y_t, [-1])

        # 添加批次维度
        ones = tf.ones_like(x_t_flat)
        sampling_grid = tf.stack([x_t_flat, y_t_flat, ones])
        sampling_grid = tf.expand_dims(sampling_grid, axis=0)
        sampling_grid = tf.tile(sampling_grid, tf.stack([num_batch, 1, 1]))

        # 应用变换
        theta = tf.cast(theta, 'float32')
        sampling_grid = tf.cast(sampling_grid, 'float32')
        batch_grids = tf.matmul(theta, sampling_grid)
        batch_grids = tf.reshape(batch_grids, [num_batch, 2, height, width])

        return batch_grids

    def _grid_sample(self, x, grid):
        # 从输入图像中采样
        num_batch, height, width, channels = x.shape
        grid_x, grid_y = tf.split(grid, num_or_size_splits=2, axis=3)

        # 对坐标进行归一化处理
        grid_x = ((grid_x + 1) / 2) * tf.cast(width, tf.float32)
        grid_y = ((grid_y + 1) / 2) * tf.cast(height, tf.float32)

        # 采样
        x_transformed = tf.map_fn(lambda z: self._bilinear_sampler(z[0], z[1], z[2]), (x, grid_x, grid_y), dtype=tf.float32)
        return x_transformed

    def _bilinear_sampler(self, img, x, y):
        # 动态获取图像尺寸
        shape = tf.shape(img)
        if tf.rank(img) == 4:
            batch, height, width, channels = shape[0], shape[1], shape[2], shape[3]
        else:
            height, width, channels = shape[0], shape[1], shape[2]
            batch = 1

        # 将x和y坐标转换为整数
        x0 = tf.cast(tf.floor(x), tf.int32)
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), tf.int32)
        y1 = y0 + 1

        # 确保坐标不超出图像边界
        x0 = tf.clip_by_value(x0, 0, width - 1)
        x1 = tf.clip_by_value(x1, 0, width - 1)
        y0 = tf.clip_by_value(y0, 0, height - 1)
        y1 = tf.clip_by_value(y1, 0, height - 1)

        # 获取像素值
        Ia = tf.gather_nd(img, tf.stack([y0, x0], axis=2))
        Ib = tf.gather_nd(img, tf.stack([y1, x0], axis=2))
        Ic = tf.gather_nd(img, tf.stack([y0, x1], axis=2))
        Id = tf.gather_nd(img, tf.stack([y1, x1], axis=2))

        # 计算插值
        x0 = tf.cast(x0, tf.float32)
        x1 = tf.cast(x1, tf.float32)
        y0 = tf.cast(y0, tf.float32)
        y1 = tf.cast(y1, tf.float32)

        wa = (x1 - x) * (y1 - y)
        wb = (x1 - x) * (y - y0)
        wc = (x - x0) * (y1 - y)
        wd = (x - x0) * (y - y0)

        # 加权求和
        output = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
        return output

# 构建带有残差连接和STN的简化版VGG模型
def build_simple_vgg_with_stn(input_shape=(224, 224, 3), num_classes=5):
    inputs = Input(shape=input_shape)

    # 加入STN层
    x = SpatialTransformer()(inputs)

    # Block 1 with residual connection
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    residual = Conv2D(32, (1, 1), strides=(2, 2), padding='same')(inputs)
    x = Add()([x, residual])
    x = Dropout(0.25)(x)  # Add Dropout after the first block

    # Block 2 with residual connection
    residual = Conv2D(64, (1, 1), strides=(2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Add()([x, residual])
    x = Dropout(0.25)(x)  # Add Dropout after the second block

    # Block 3 with residual connection
    residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Add()([x, residual])
    x = Dropout(0.25)(x)  # Add Dropout after the third block

    # Classification block
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)  # Add Dropout before the final classification layer
    x = Dense(512, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, x)
    return model

# 构建并编译模型
model = build_simple_vgg_with_stn()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 打印模型结构
model.summary()

# 数据预处理
train_dir = '/content/drive/My Drive/data/train'
validation_dir = '/content/drive/My Drive/data/validation'

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=20,
    class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=20,
    class_mode='categorical')

# 训练模型
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,
    steps_per_epoch=100,
    validation_steps=50
)

# 可视化训练过程
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

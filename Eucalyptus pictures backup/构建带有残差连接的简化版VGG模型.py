#我想在这个文件里加STN，总是出现通道数的问题



import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dropout

# 构建带有残差连接的简化版VGG模型
def build_simple_vgg_with_residuals(input_shape=(224, 224, 3), num_classes=5):
    inputs = Input(shape=input_shape)

    # Block 1 with residual connection
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
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
    x = Dropout(0.25)(x)  # Add Dropout after the first block

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





# 数据预处理
train_dir = '/content/drive/My Drive/data/train'
validation_dir = '/content/drive/My Drive/data/validation'

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=20, #这里改了
    class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=20,
    class_mode='categorical')

# # 构建并编译模型
# model = build_simple_vgg()
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 构建并编译模型
model = build_simple_vgg_with_residuals()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])




# 训练模型
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,  # 或者选择的其他epoch数
    steps_per_epoch=100,  # 根据数据集调整
    validation_steps=25  # 根据数据集调整
)


# 可视化训练过程
plt.figure(figsize=(12, 4))

# 绘制准确率曲线
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# 绘制损失曲线
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam  # 或者 from tensorflow.keras.optimizers.legacy import Adam（对于 M1/M2 Mac）
import matplotlib.pyplot as plt
from PIL import Image
import os

# 检查图像文件是否损坏
def check_images(s_dir, ext_list):
    bad_images = []
    for folder in os.listdir(s_dir):
        folder_path = os.path.join(s_dir, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if file.split('.')[-1] in ext_list:
                    try:
                        img = Image.open(file_path)
                        img.verify()
                    except (IOError, SyntaxError):
                        print('Bad file:', file_path)
                        bad_images.append(file_path)
    return bad_images

# 检查训练和验证集中的图像
print("Checking images in training set...")
bad_train_images = check_images('./图片备份/train', ['jpg', 'png'])
print(f"Found {len(bad_train_images)} bad images in training set")

print("Checking images in validation set...")
bad_val_images = check_images('./图片备份/validation', ['jpg', 'png'])
print(f"Found {len(bad_val_images)} bad images in validation set")

# 数据预处理
train_dir = './图片备份/train'
validation_dir = './图片备份/validation'

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

# 构建 VGG16 模型
base_model = VGG16(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
for layer in base_model.layers:
    layer.trainable = False

x = Flatten()(base_model.output)
x = Dense(5, activation='softmax')(x)  # 假设有5个类别

model = Model(base_model.input, x)

# 编译模型
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,
    steps_per_epoch=100,
    validation_steps=50)

# 可视化
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy over Epochs')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss over Epochs')

plt.show()

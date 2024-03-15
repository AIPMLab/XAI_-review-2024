import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import EfficientNetB0
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import cv2
import os
import random
import shap
import matplotlib.pyplot as plt
import warnings
import matplotlib.cm as cm
from tensorflow.keras.models import Model
from sklearn.metrics import classification_report, confusion_matrix
warnings.filterwarnings("ignore")

# 确认 TensorFlow 能够识别 GPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# 为 GPU 设置内存增长
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)


def load_and_preprocess_data(datasetfolder):
    # 训练集数据增强
    ge = ImageDataGenerator(rescale=1/255,
                            rotation_range=10,
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            fill_mode='constant',
                            validation_split=0.2,  # 保留20%的数据用于验证
                            horizontal_flip=True,
                            vertical_flip=True,
                            zoom_range=0.1
                            )

    # 为训练数据创建一个数据生成器实例
    dataflowtraining = ge.flow_from_directory(directory=datasetfolder,
                                              target_size=(224, 224),
                                              color_mode='rgb',
                                              batch_size=32,
                                              shuffle=True,
                                              subset='training')

    # 验证集不应用数据增强，只进行归一化
    ge_validation = ImageDataGenerator(rescale=1/255, validation_split=0.2)  # 只有归一化处理

    # 为验证数据创建一个数据生成器实例
    dataflowvalidation = ge_validation.flow_from_directory(directory=datasetfolder,
                                                            target_size=(224, 224),
                                                            color_mode='rgb',
                                                            batch_size=32,
                                                            shuffle=True,
                                                            subset='validation')
    return dataflowtraining, dataflowvalidation

from tensorflow.keras.callbacks import ReduceLROnPlateau

# 学习率调度器
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,  # 当验证集损失停止改善时，学习率减少为原来的0.1
    patience=5,  # 如果5个epoch后看不到验证损失的改善，则减少学习率
    verbose=1
)

callbacks = [reduce_lr]

def build_model(num_classes=4):
    basemodel = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = tf.keras.layers.Flatten()(basemodel.output)
    x = tf.keras.layers.Dropout(0.5)(x)  # 降低Dropout比例，原来是0.7
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)  # 降低Dropout比例，原来是0.5
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=basemodel.input, outputs=x)
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # 调整学习率
                  metrics=['accuracy', tf.keras.metrics.Precision(name='precision'),
                           tf.keras.metrics.Recall(name='recall')])
    return model



def plot_history(hist, save_dir='training_plots'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Plot training & validation accuracy values
    plt.figure(figsize=(10, 5))
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(save_dir, 'model_accuracy.png'))
    plt.show()
    plt.close()

    # Plot training & validation loss values
    plt.figure(figsize=(10, 5))
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(save_dir, 'model_loss.png'))
    plt.show()
    plt.close()

from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def evaluate_model(model, dataset_folder):
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        dataset_folder,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False)

    Y_pred = model.predict(test_generator, test_generator.samples // test_generator.batch_size + 1)
    y_pred = np.argmax(Y_pred, axis=1)
    print('Confusion Matrix')
    print(confusion_matrix(test_generator.classes, y_pred))
    print('Classification Report')
    target_names = list(test_generator.class_indices.keys())
    print(classification_report(test_generator.classes, y_pred, target_names=target_names))


def preprocess_image(img_path, target_size=(224, 224)):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0) / 255.0  # 此处确保图像值在0到1之间
    return img



# 定义 train_model 函数
def train_model(model, train_generator, validation_generator):
    history = model.fit(
        train_generator,
        epochs=100,
        validation_data=validation_generator,
        verbose=1,
        callbacks=callbacks
    )


    return history

def main():
    data_path = "C:\\Users\\PS\\Desktop\\wen project\\eye disease\\output\\train"
    train_generator, validation_generator = load_and_preprocess_data(data_path)
    from tensorflow.keras.mixed_precision import experimental as mixed_precision

    model = build_model()
    for i, layer in enumerate(model.layers):
        print(i, layer.name)
    history = train_model(model, train_generator, validation_generator)

    plot_history(history)

    # 模型性能评估
    test_data_path = "C:\\Users\\PS\\Desktop\\wen project\\eye disease\\output\\test"
    evaluate_model(model, test_data_path)

    # 模型保存路径和文件名
    model_save_dir = "C:\\Users\\PS\\Desktop\\wen project"
    model_file_name = "deneye_model"  # 你可以根据需要自定义文件名

    # 确保保存目录存在
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    #
    # # 完整的模型保存路径（SavedModel格式）
    model_save_path_tf = os.path.join(model_save_dir, model_file_name)

    # 保存为TensorFlow的SavedModel格式
    model.save(model_save_path_tf, save_format='tf')


if __name__ == "__main__":
    main()



from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shap
import cv2
import matplotlib.pyplot as plt
import os

def preprocess_image(img_path, target_size=(224, 224)):
    """
    加载和预处理图像
    """
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0) / 255.0  # 归一化
    return img

def explain_shap(model, img_array, class_indices, background_images):
    # 创建 Deep SHAP 解释器
    explainer = shap.GradientExplainer(model, background_images)

    # 计算 SHAP 值
    shap_values = explainer.shap_values(img_array)

    # 对于每个类别，可视化 SHAP 值
    for i, class_index in enumerate(class_indices):
        # 注意：根据你的需求调整可视化代码
        shap.image_plot(shap_values[class_index], img_array[i:i+1])

labels = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
# labels = ['COVID19', 'NORMAL', 'PNEUMONIA']
# labels = ['0_normal', '1_ulcerative_colitis', '2_polyps', '3_esophagitis']
# labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
# labels = ['adenocarcinoma', 'large.cell.carcinoma', 'normal', 'squamous.cell.carcinoma']
def load_and_preprocess_data(datasetfolder):
    ge = ImageDataGenerator(rescale=1/255,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        fill_mode='constant',
                        validation_split=0.2,
                        horizontal_flip=True,
                        vertical_flip=True,
                        zoom_range=0.1
                        )
    dataflowtraining = ge.flow_from_directory(directory=datasetfolder,
                                              target_size=(224, 224),
                                              color_mode='rgb',
                                              batch_size=32,
                                              shuffle=True,
                                              subset='training')
    dataflowvalidation = ge.flow_from_directory(directory=datasetfolder,
                                                target_size=(224, 224),
                                                color_mode='rgb',
                                                batch_size=32,
                                                shuffle=True,
                                                subset='validation')
    return dataflowtraining, dataflowvalidation
# def load_and_preprocess_data(datasetfolder):
#     # 训练集数据增强
#     ge = ImageDataGenerator(rescale=1/255,
#                             rotation_range=10,
#                             width_shift_range=0.1,
#                             height_shift_range=0.1,
#                             fill_mode='constant',
#                             validation_split=0.2,  # 保留20%的数据用于验证
#                             horizontal_flip=True,
#                             vertical_flip=True,
#                             zoom_range=0.1
#                             )
#
#     # 为训练数据创建一个数据生成器实例
#     dataflowtraining = ge.flow_from_directory(directory=datasetfolder,
#                                               target_size=(224, 224),
#                                               color_mode='rgb',
#                                               batch_size=32,
#                                               shuffle=True,
#                                               subset='training')
#
#     # 验证集不应用数据增强，只进行归一化
#     ge_validation = ImageDataGenerator(rescale=1/255, validation_split=0.2)  # 只有归一化处理
#
#     # 为验证数据创建一个数据生成器实例
#     dataflowvalidation = ge_validation.flow_from_directory(directory=datasetfolder,
#                                                             target_size=(224, 224),
#                                                             color_mode='rgb',
#                                                             batch_size=32,
#                                                             shuffle=True,
#                                                             subset='validation')
#     return dataflowtraining, dataflowvalidation

def main():
    # 加载预训练模型
    # 假设 model_path 指向的是 SavedModel 格式模型的目录
    model_path = 'C:\\Users\\PS\\Desktop\\wen project\\b0eye_model'
    model = tf.keras.models.load_model(model_path)

    # 加载和预处理特定的图像
    # img_paths = [
    #     'C:\\Users\\PS\\Desktop\\wen project\\eye disease\\output\\test\\glaucoma\\_60_3287251.jpg',
    #     'C:\\Users\\PS\\Desktop\\wen project\\eye disease\\output\\test\\diabetic_retinopathy\\11584_right.jpeg',
    #     'C:\\Users\\PS\\Desktop\\wen project\\eye disease\\output\\test\\cataract\\2168_left.jpg',
    #     'C:\\Users\\PS\\Desktop\\wen project\\eye disease\\output\\test\\cataract\\_57_8463167.jpg'
    # ]
    img_paths = ["C:\\Users\\PS\\Desktop\\wen project\\eye disease\\output\\test\\normal\\3097_left.jpg"]
    # "C:\\Users\\PS\\Desktop\\wen project\\covid-19\\Data\\test\\PNEUMONIA\\PNEUMONIA(3518).jpg"
    #              "C:\\Users\\PS\Desktop\\wen project\\covid-19\\Data\\test\\NORMAL\\NORMAL(1414).jpg"
    #              "C:\\Users\\PS\\Desktop\\wen project\\covid-19\\Data\\test\\PNEUMONIA\\PNEUMONIA(3574).jpg"
    #              "C:\\Users\\PS\\Desktop\\wen project\\covid-19\\Data\\test\\COVID19\\COVID19(549).jpg"
    # colon data
    # img_paths = ["C:\\Users\\PS\\Desktop\\wen project\\colon\\test\\3_esophagitis\\test_esophagitis_ (128).jpg"
    #              ]
    # "C:\\Users\\PS\\Desktop\\wen project\\colon\\test\\0_normal\\test_normal_ (93).jpg"
    # "C:\\Users\\PS\\Desktop\\wen project\\colon\\test\\1_ulcerative_colitis\\test_ulcer_ (147).jpg"
    # "C:\\Users\\PS\\Desktop\\wen project\\colon\\test\\2_polyps\\test_polyps_ (116).jpg"
    # "C:\\Users\\PS\\Desktop\\wen project\\colon\\test\\3_esophagitis\\test_esophagitis_ (128).jpg"
    # brain tumor
    # img_paths = ["C:\\Users\\PS\\Desktop\\wen project\\brain tumor dataset\\Testing\\pituitary\\Te-pi_0245.jpg"]
    # "C:\\Users\\PS\\Desktop\\wen project\\brain tumor dataset\\Testing\\glioma\\Te-gl_0208.jpg"
    #              "C:\\Users\\PS\\Desktop\\wen project\\brain tumor dataset\\Testing\\meningioma\\Te-me_0221.jpg"
    #              "C:\\Users\\PS\\Desktop\\wen project\\brain tumor dataset\\Testing\\notumor\\Te-no_0255.jpg"
    #              "C:\\Users\\PS\\Desktop\\wen project\\brain tumor dataset\\Testing\\pituitary\\Te-pi_0245.jpg"

    # ct cancer data
    # img_paths = ["C:\\Users\\PS\\Desktop\\wen project\\cancer ct\\test\\squamous.cell.carcinoma\\000137.png"
    #               ]
    # "C:\\Users\\PS\\Desktop\\wen project\\cancer ct\\test\\adenocarcinoma\\000115 (8).png"
    # "C:\\Users\\PS\\Desktop\\wen project\\cancer ct\\test\\large.cell.carcinoma\\000158.png"
    # "C:\\Users\\PS\\Desktop\\wen project\\cancer ct\\test\\normal\\10 (2) - Copy.png"
    # "C:\\Users\\PS\\Desktop\\wen project\\cancer ct\\test\\squamous.cell.carcinoma\\000137.png"
    img_arrays = np.vstack([preprocess_image(path) for path in img_paths])

    preds = model.predict(img_arrays)
    pred_indices = np.argmax(preds, axis=1)
    for i, pred_index in enumerate(pred_indices):
        print(f"Image {i + 1} predicted class: {labels[pred_index]} with confidence {preds[i][pred_index]:.2f}")

    # 使用一组背景图像初始化 SHAP 解释器
    data_path = "C:\\Users\\PS\\Desktop\\wen project\\eye disease\\output\\train"
    train_generator, _ = load_and_preprocess_data(data_path)
    background_images, _ = next(train_generator)

    # SHAP 解释器初始化及解释
    explain_shap(model, img_arrays, pred_indices, background_images)

if __name__ == "__main__":
    main()

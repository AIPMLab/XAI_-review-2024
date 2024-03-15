import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 加载模型，并确保模型处于推理模式
# model = keras.models.load_model('C:\\Users\\PS\\Desktop\\wen project\\resnet eye.h5')
# model = keras.models.load_model('C:\\Users\\PS\\Desktop\\wen project\\dense eye.h5')
model = keras.models.load_model('C:\\Users\\PS\\Desktop\\wen project\\deneye_model')
model.trainable = False

# 设定最后的卷积层名称
last_conv_layer_name = 'conv5_block16_concat' # ResNet50--conv5_block3_out,  densenet121--conv5_block16_concat,  efficientnetb3--top_conv

# 标签列表b0
# labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
labels = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
# labels = ['adenocarcinoma', 'large.cell.carcinoma', 'normal', 'squamous.cell.carcinoma']
# labels = ['COVID19', 'NORMAL', 'PNEUMONIA']
# labels = ['0_normal', '1_ulcerative_colitis', '2_polyps', '3_esophagitis']
def preprocess_image(img_path, target_size=(224, 224)):
    img = keras.preprocessing.image.load_img(img_path, target_size=target_size)
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0) / 255.0
    return img

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # 获取模型的最后一个卷积层
    last_conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = Model(inputs=[model.inputs], outputs=[last_conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.matmul(conv_outputs, pooled_grads[..., tf.newaxis])
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(img, heatmap, alpha=0.4, threshold=0.5):
    # 将heatmap数据类型转换为float32
    heatmap = heatmap.astype('float32')

    # 应用阈值，以突出显示重要区域
    heatmap = np.maximum(heatmap, threshold)
    heatmap = heatmap / np.max(heatmap)

    # 调整heatmap大小以匹配原始图像的大小
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # 将heatmap标准化到0-255范围内，并转换为整数
    heatmap = np.uint8(255 * heatmap)

    # 应用颜色映射
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)

    # 创建一个只在热图有活动区域的掩码
    active_regions = np.any(heatmap > 0, axis=-1)

    # 创建一个只有活动区域被颜色映射覆盖的图像
    img_with_heatmap = np.zeros_like(img)
    img_with_heatmap[active_regions] = heatmap[active_regions]

    # 叠加heatmap到原始图像
    superimposed_img = img_with_heatmap * alpha + img * (1 - alpha)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')

    # 显示图像
    plt.imshow(superimposed_img)
    plt.title("Grad-CAM")
    plt.axis('off')
    plt.show()


def main(img_paths):
    for img_path in img_paths:
        img_array = preprocess_image(img_path)
        preds = model.predict(img_array)
        pred_index = np.argmax(preds[0])
        print(f"Predicted class: {labels[pred_index]} with confidence {preds[0][pred_index]:.2f}")

        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        display_gradcam(img, heatmap, alpha=0.4, threshold=0.2)
# eye test data
# img_paths = ['C:\\Users\\PS\\Desktop\\wen project\\eye disease\\output\\test\\glaucoma\\_60_3287251.jpg',
#               'C:\\Users\\PS\\Desktop\\wen project\\eye disease\\output\\test\\diabetic_retinopathy\\11584_right.jpeg',
#               'C:\\Users\\PS\\Desktop\\wen project\\eye disease\\output\\test\\cataract\\2168_left.jpg',
#                'C:\\Users\\PS\\Desktop\\wen project\\eye disease\\output\\test\\cataract\\_57_8463167.jpg']
img_paths = ["C:\\Users\\PS\\Desktop\\wen project\\eye disease\\output\\test\\normal\\3097_left.jpg"]
# brain tumor data
# img_paths = ["C:\\Users\\PS\\Desktop\\wen project\\brain tumor dataset\\Testing\\glioma\\Te-gl_0208.jpg",
#              "C:\\Users\\PS\\Desktop\\wen project\\brain tumor dataset\\Testing\\meningioma\\Te-me_0221.jpg",
#              "C:\\Users\\PS\\Desktop\\wen project\\brain tumor dataset\\Testing\\notumor\\Te-no_0255.jpg",
#              "C:\\Users\\PS\\Desktop\\wen project\\brain tumor dataset\\Testing\\pituitary\\Te-pi_0245.jpg"]

# cancer ct data
# img_paths = ["C:\\Users\\PS\\Desktop\\wen project\\cancer ct\\test\\adenocarcinoma\\000115 (8).png",
#              "C:\\Users\\PS\\Desktop\\wen project\\cancer ct\\test\\large.cell.carcinoma\\000158.png",
#              "C:\\Users\\PS\\Desktop\\wen project\\cancer ct\\test\\normal\\10 (2) - Copy.png",
#              "C:\\Users\\PS\\Desktop\\wen project\\cancer ct\\test\\squamous.cell.carcinoma\\000137.png"]


# covid19 data
# img_paths = ["C:\\Users\\PS\\Desktop\\wen project\\covid-19\\Data\\test\\PNEUMONIA\\PNEUMONIA(3518).jpg",
#              "C:\\Users\\PS\Desktop\\wen project\\covid-19\\Data\\test\\NORMAL\\NORMAL(1414).jpg",
#              "C:\\Users\\PS\\Desktop\\wen project\\covid-19\\Data\\test\\PNEUMONIA\\PNEUMONIA(3574).jpg",
#              "C:\\Users\\PS\\Desktop\\wen project\\covid-19\\Data\\test\\COVID19\\COVID19(549).jpg"]

# colon data
# img_paths = ["C:\\Users\\PS\\Desktop\\wen project\\colon\\test\\0_normal\\test_normal_ (93).jpg",
#              "C:\\Users\\PS\\Desktop\\wen project\\colon\\test\\1_ulcerative_colitis\\test_ulcer_ (147).jpg",
#              "C:\\Users\\PS\\Desktop\\wen project\\colon\\test\\2_polyps\\test_polyps_ (116).jpg",
#              "C:\\Users\\PS\\Desktop\\wen project\\colon\\test\\3_esophagitis\\test_esophagitis_ (128).jpg"]

main(img_paths)

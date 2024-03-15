from tensorflow.keras.models import load_model
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def preprocess_image(img_path, target_size=(224, 224)):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0) / 255.0  # 此处确保图像值在0到1之间
    return img

def integrated_gradients(img_array, model, baseline=None, steps=50):
    """
    计算 Integrated Gradients。

    :param img_array: 原始输入图像的数组。
    :param model: Keras 模型。
    :param baseline: 用作起点的基线图像。如果为 None，则使用全零图像。
    :param steps: 积分步数。
    :return: Integrated Gradients 数组。
    """
    # 如果没有提供基线图像，使用全零图像
    if baseline is None:
        # baseline = np.zeros(img_array.shape)
        baseline = np.zeros(img_array.shape).astype(np.float32)

    # 初始化基线和图像之间的步长
    step_list = [baseline + (float(i) / steps) * (img_array - baseline) for i in range(steps + 1)]

    # 将步长列表转换为 TensorFlow 张量
    step_list_tensor = [tf.convert_to_tensor(step, dtype=tf.float32) for step in step_list]

    # 使用 tf.concat 而不是 tf.stack 来保持形状一致
    steps_tensor = tf.concat(step_list_tensor, axis=0)

    # 计算每一步的梯度
    with tf.GradientTape() as tape:
        # 监控整个步骤张量
        tape.watch(steps_tensor)
        predictions = model(steps_tensor)

    gradients = tape.gradient(predictions, steps_tensor)

    # 计算积分的逼近值
    avg_gradients = np.average([grad.numpy() for grad in gradients], axis=0)

    # 计算 Integrated Gradients
    integrated_gradients = (img_array - baseline) * avg_gradients

    # 归一化处理
    ig_norm = integrated_gradients - integrated_gradients.min()
    ig_norm /= ig_norm.max()

    # Return the normalized integrated gradients
    return ig_norm

def plot_integrated_gradients(ig_attributions, original_img):
    # Resize ig_attributions to match the original image size
    ig_attributions_resized = tf.image.resize(ig_attributions, (original_img.shape[0], original_img.shape[1]))

    # Normalize the attributions
    ig_attributions_resized = (ig_attributions_resized - np.min(ig_attributions_resized)) / (
                np.max(ig_attributions_resized) - np.min(ig_attributions_resized))

    # Multiply by the original image (element-wise multiplication)
    highlighted_img = (ig_attributions_resized * original_img).numpy().astype(np.uint8)

    # 展示原始图像、IG 属性图以及强调特征的图像
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(original_img)
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    axs[1].imshow(ig_attributions)
    axs[1].set_title('Integrated Gradients')
    axs[1].axis('off')

    axs[2].imshow(highlighted_img)
    axs[2].set_title('Highlighted Features')
    axs[2].axis('off')

    plt.show()

# 加载模型进行预测和解释
def main():
    # 假设模型文件位于与此脚本相同的目录
    model_path = 'C:\\Users\\PS\\Desktop\\wen project\\b0eye_model'
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        raise FileNotFoundError(f"{model_path} not found.")



    # 随机选择不同类别的图片或指定一个图片路径
    # eye data
    # img_path = "C:\\Users\\PS\\Desktop\\wen project\\eye disease\\output\\test\\glaucoma\\_60_3287251.jpg"
    # img_path = "C:\\Users\\PS\\Desktop\\wen project\\eye disease\\output\\test\\diabetic_retinopathy\\11584_right.jpeg"
    img_path = "C:\\Users\\PS\\Desktop\\wen project\\eye disease\\output\\test\\normal\\3097_left.jpg"
    # img_path = "C:\\Users\\PS\\Desktop\\wen project\\eye disease\\output\\test\\cataract\\_57_8463167.jpg"

    # covid data
    # img_path = "C:\\Users\\PS\\Desktop\\wen project\\covid-19\\Data\\test\\PNEUMONIA\\PNEUMONIA(3518).jpg"
    # img_path = "C:\\Users\\PS\Desktop\\wen project\\covid-19\\Data\\test\\NORMAL\\NORMAL(1414).jpg"
    # img_path = "C:\\Users\\PS\\Desktop\\wen project\\covid-19\\Data\\test\\PNEUMONIA\\PNEUMONIA(3574).jpg"
    # img_path = "C:\\Users\\PS\\Desktop\\wen project\\covid-19\\Data\\test\\COVID19\\COVID19(549).jpg"

    # colon data
    # img_path = "C:\\Users\\PS\\Desktop\\wen project\\colon\\test\\0_normal\\test_normal_ (93).jpg"
    # img_path ="C:\\Users\\PS\\Desktop\\wen project\\colon\\test\\1_ulcerative_colitis\\test_ulcer_ (147).jpg"
    # img_path ="C:\\Users\\PS\\Desktop\\wen project\\colon\\test\\2_polyps\\test_polyps_ (116).jpg"
    # img_path = "C:\\Users\\PS\\Desktop\\wen project\\colon\\test\\3_esophagitis\\test_esophagitis_ (128).jpg"

    # ct cancer data
    # img_path = "C:\\Users\\PS\\Desktop\\wen project\\cancer ct\\test\\adenocarcinoma\\000115 (8).png"
    # img_path = "C:\\Users\\PS\\Desktop\\wen project\\cancer ct\\test\\large.cell.carcinoma\\000158.png"
    # img_path = "C:\\Users\\PS\\Desktop\\wen project\\cancer ct\\test\\normal\\10 (2) - Copy.png"
    # img_path = "C:\\Users\\PS\\Desktop\\wen project\\cancer ct\\test\\squamous.cell.carcinoma\\000137.png"

    # brain tumor data
    # img_path = "C:\\Users\\PS\\Desktop\\wen project\\brain tumor dataset\\Testing\\glioma\\Te-gl_0208.jpg"
    # img_path = "C:\\Users\\PS\\Desktop\\wen project\\brain tumor dataset\\Testing\\meningioma\\Te-me_0221.jpg"
    # img_path = "C:\\Users\\PS\\Desktop\\wen project\\brain tumor dataset\\Testing\\notumor\\Te-no_0255.jpg"
    # img_path = "C:\\Users\\PS\\Desktop\\wen project\\brain tumor dataset\\Testing\\pituitary\\Te-pi_0245.jpg"

    img_array = preprocess_image(img_path)
    img_tensor = tf.convert_to_tensor(preprocess_image(img_path), dtype=tf.float32)

    # 应用 Integrated Gradients
    ig_attributions = integrated_gradients(img_array, model)

    # Check if ig_attributions is not None
    if ig_attributions is not None:
        # 可视化 Integrated Gradients
        plt.figure()
        plt.imshow(ig_attributions[0])
        plt.title("Integrated Gradients")
        plt.axis('off')
        plt.show()
    else:
        print("Integrated Gradients returned None. Check your function implementation.")

    original_img = cv2.imread(img_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)  # 转换为 RGB

    if ig_attributions is not None:
        # 这里调用新的可视化函数
        plot_integrated_gradients(ig_attributions[0], original_img)
    else:
        print("Integrated Gradients returned None. Check your function implementation.")

if __name__ == "__main__":
    main()

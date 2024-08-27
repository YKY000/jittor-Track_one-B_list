import jittor as jt
from PIL import Image, ImageFilter, ImageEnhance, ImageOps, ImageDraw
import numpy as np
from jittor.transform import ColorJitter, RandomHorizontalFlip, RandomAffine, Gray, vflip



def Blur(image):
    """
    对给定的图像应用高斯模糊处理。

    该函数通过随机选择一个3到8之间的核大小,对图像进行高斯模糊。这种模糊处理可以用于减少图像的噪声,
    或者平滑图像,使其看起来更柔和。

    参数:
    image: PIL库中的Image对象,表示要进行模糊处理的图像。

    返回值:
    返回应用了高斯模糊处理的图像。
    """
    # 随机选择高斯模糊的核大小,范围在3到8之间,以增加处理的随机性。
    kernel_size = np.random.randint(3, 8)
    # 应用高斯模糊滤镜,并返回处理后的图像。
    return image.filter(ImageFilter.GaussianBlur(kernel_size))


def cutout(image, padding=0.2):
    """
    对给定的图像应用CutOut数据增广。

    参数:
    image: PIL库中的Image对象,表示要进行CutOut处理的图像。
    padding: 控制CutOut矩形区域大小的参数,范围在0到1之间。

    返回值:
    返回应用了CutOut处理的图像。
    """
    w, h = image.size
    x, y = np.random.randint(0, w, size=2)
    length = int(padding * min(w, h))

    # 将矩形区域设置为黑色
    cut_image = image.copy()
    ImageDraw.Draw(cut_image).rectangle([(x, y), (x + length, y + length)], fill=0)

    return cut_image


def cutmix(image, alpha=0.4):
    """
    对给定的图像应用CutMix数据增广。

    参数:
    image: PIL库中的Image对象,表示要进行CutMix处理的图像。
    alpha: 控制CutMix矩形区域大小的参数,范围在0到1之间。

    返回值:
    返回应用了CutMix处理的图像。
    """
    # 随机生成矩形区域
    w, h = image.size
    x, y = np.random.randint(0, w, size=2)
    width = int(alpha * w)
    height = int(alpha * h)

    # 将矩形区域设置为黑色
    cut_image = image.copy()
    ImageDraw.Draw(cut_image).rectangle([(x, y), (x + width, y + height)], fill=0)

    # 将原图像与矩形区域被黑色覆盖的图像按比例混合
    mixed_image = Image.blend(image, cut_image, (1 - alpha))

    return mixed_image


def adjust_contrast(image, contrast_factor):
    """提升对比度"""
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(contrast_factor)


def color_jitter(image, brightness=0.2, contrast=0.3, saturation=0.4, hue=0.1):
    """
    对给定的图像应用ColorJitter数据增广。

    参数:
    image: PIL库中的Image对象,表示要进行ColorJitter处理的图像。
    brightness: 控制亮度的调整参数。
    contrast: 控制对比度的调整参数。
    saturation: 控制饱和度的调整参数。
    hue: 控制色相的调整参数。

    返回值:
    返回应用了ColorJitter处理的图像。
    """
    transform = ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
    return transform(image)

def Random_Affine(image):
    """
    对给定的图像应用随机仿射变换数据增广。

    参数:
    image: PIL库中的Image对象,表示要进行Random_Affine处理的图像。

    """
    transform = RandomAffine(degrees=45)
    return transform(image)

def Gray_img(image):
    """
    对给定的图像应用灰度图变换数据增广。

    """
    transform = Gray()
    return transform(image)

def Vflip(image):
    """
    对给定的图像应用垂直翻转变换数据增广。

    """
    return vflip(image)

def Random_HorizontalFlip_brightness(image):
    """
    对给定的图像应用随机水平翻转并进行亮度调整数据增广。

    """
    transform = RandomHorizontalFlip(0.6)
    img  =  transform(image)
    img = jt.transform.adjust_brightness(img, 1.3)
    img = jt.transform.adjust_contrast(img, 1.3)
    return img


def apply_data_augmentation(image, augmentation_func, transform=None, model=None):
    """
    对给定的图像应用数据增广方法。

    参数:
    image: PIL库中的Image对象,表示要进行数据增广处理的图像。
    augmentation_func: 数据增广函数,如Blur、cutout、cutmix等。

    返回值:
    返回应用了数据增广处理的图像的特征向量。
    """
    augmented_image = augmentation_func(image)
    preprocessed_augmented_image = transform(augmented_image).unsqueeze(0)
    return preprocessed_augmented_image



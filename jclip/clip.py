import hashlib
import os
import numpy as np
import urllib
import warnings
from typing import Union, List

import jittor as jt
from tqdm import tqdm

from .model import build_model
from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from PIL import Image
from jittor.transform import CenterCrop, ImageNormalize, Compose, _setup_size, to_pil_image, resize

__all__ = ["available_models", "load", "tokenize"]
_tokenizer = _Tokenizer()

_MODELS = {
    "RN50":
    "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101":
    "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4":
    "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16":
    "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64":
    "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32":
    "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16":
    "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14":
    "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px":
    "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}


def _download(url: str, root: str):
    """
    下载给定URL的文件并将其保存到指定的根目录中。

    如果文件已经下载，则会检查文件的SHA256校验和是否与预期值匹配。如果校验和不匹配，则会重新下载文件。

    参数:
    url: str - 文件的URL。
    root: str - 文件将被保存的目录。

    返回:
    str - 下载文件的完整路径。
    """
    os.makedirs(root, exist_ok=True)
    # 从URL中提取文件名
    filename = os.path.basename(url)

    # 从URL中提取预期的SHA256校验和
    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(
            f"{download_target} exists and is not a regular file")

    # 如果目标文件已存在，检查其SHA256校验和是否与预期值匹配
    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target,
                               "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(
                f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file"
            )

    # 打开URL和目标文件，准备下载
    with urllib.request.urlopen(url) as source, open(download_target,
                                                     "wb") as output:
        # 使用tqdm库显示下载进度条
        with tqdm(total=int(source.info().get("Content-Length")),
                  ncols=80,
                  unit='iB',
                  unit_scale=True,
                  unit_divisor=1024) as loop:
            # 循环读取和写入数据，直到下载完成
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    # 下载完成后，再次检查SHA256校验和
    if hashlib.sha256(open(download_target,
                           "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError(
            "Model has been downloaded but the SHA256 checksum does not not match"
        )
    # 返回下载文件的完整路径
    return download_target


def _convert_image_to_rgb(image):
    """
    将图像转换为RGB格式。

    该函数接受一个图像对象作为输入，将其转换为RGB格式后返回。

    参数:
    image: PIL库中的Image对象，表示待转换的图像。

    返回值:
    返回转换后的Image对象，其颜色模式为RGB。
    """
    return image.convert("RGB")


def to_tensor(data):
    """
    将给定数据转换为计图（Jittor）变量。

    参数:
    - data: 需要转换的数据，可以是各种数据类型，如列表、数组或标量。

    返回值:
    - jt.Var: 转换后的计图变量，用于计图框架中的计算和操作。
    """
    return jt.Var(data)


class ImageToTensor(object):
    """
        将输入的图像数据转换为张量格式。

        该类的主要作用是处理输入的图像数据，确保其格式符合进一步处理（如模型输入）的要求。
    """
    def __call__(self, input):
        # 将输入的图像数据转换为Numpy数组格式
        input = np.asarray(input)
        # 如果图像数据的维度少于3（例如，灰度图像），则添加一个维度使其满足要求
        if len(input.shape) < 3:
            input = np.expand_dims(input, -1)
        return to_tensor(input)


class Resize:
    """
        图像大小调整类，用于将图像resize到指定的大小。

        Attributes:
            size (int or tuple): 需要调整到的大小。可以是一个整数，表示短边的长度；也可以是一个元组，表示(width, height)。
            mode (int): resize操作使用的插值方法，默认为BILINEAR（双线性插值）。
    """
    def __init__(self, size, mode=Image.BILINEAR):
        if isinstance(size, int):
            self.size = size
        else:
            self.size = _setup_size(
                size,
                error_msg="If size is a sequence, it should have 2 values")
        self.mode = mode

    def __call__(self, img: Image.Image):
        """
        使类实例可被直接调用，接收一个图像对象，返回调整大小后的图像。

        Args:
            img (Image.Image): PIL库中的图像对象。

        Returns:
            Image.Image: 调整大小后的图像对象。
        """
        if not isinstance(img, Image.Image):
            img = to_pil_image(img)
        # 检查self.size是否为整数，如果是，进行图像尺寸调整
        if isinstance(self.size, int):
            w, h = img.size
            # 确定图像的短边和长边
            short, long = (w, h) if w <= h else (h, w)
            if short == self.size:
                return img

            new_short, new_long = self.size, int(self.size * long / short)
            # 根据原始图像的宽高比例，确定新图像的宽度和高度
            new_w, new_h = (new_short, new_long) if w <= h else (new_long,
                                                                 new_short)
            # 设置新图像的尺寸
            size = (new_h, new_w)
        return resize(img, size, self.mode)


def _transform(n_px):
    """
    构造图像变换序列，用于模型训练或数据预处理。

    这个函数返回一个图像处理序列，包括调整图像大小、中心裁剪、转换为RGB格式、归一化和转换为张量等步骤。
    这些步骤对于准备图像数据以供深度学习模型使用是非常关键的。

    参数:
    n_px: int - 指定图像的大小，经过变换后，图像的长和宽都将调整为这个值。

    返回:
    Compose - 图像变换的序列，包含了多个图像处理步骤。
    """
    return Compose([
        # 调整图像大小，使用双线性插值
        Resize(n_px, mode=Image.BICUBIC),
        # 中心裁剪图像，将图像转换为RGB格式
        CenterCrop(n_px), _convert_image_to_rgb,
        # 对图像进行归一化，使用预定义的均值和标准差
        ImageNormalize((0.48145466, 0.4578275, 0.40821073),
                       (0.26862954, 0.26130258, 0.27577711)),
        # 将图像数据转换为张量格式
        ImageToTensor()
    ])


def available_models() -> List[str]:
    """
    获取所有可用的CLIP模型名称。

    返回：
        模型名称列表。
    """
    return list(_MODELS.keys())


def load(name, download_root=None):
    """
       加载CLIP模型。

       根据提供的模型名称，尝试从预定义的模型列表中加载模型。如果模型名称不存在于列表中，
       但名称对应的路径指向一个文件，则尝试从该文件加载模型。如果既不在列表中，也不是有效文件路径，
       则抛出运行时错误。

       参数:
       - name: 模型的名称或模型文件的路径。
       - download_root: 下载模型时的根目录。如果未指定，默认为用户缓存目录下的`.cache/clip`。

       返回:
       - model: 加载后的模型实例。
       - transform: 转换为序列的转换函数
    """
    if name in _MODELS:
        model_path = _download(
            _MODELS[name], download_root
            or os.path.expanduser("~/.cache/clip"))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(
            f"Model {name} not found; available models = {available_models()}")

    # with open(model_path, 'rb') as opened_file:
    state_dict = jt.load(model_path)
    # 根据状态字典构建模型实例
    model = build_model(state_dict)
    return model, _transform(model.visual.input_resolution)


def tokenize(texts: Union[str, List[str]],
             context_length: int = 77,
             truncate: bool = False):
    """
       对文本进行分词处理，将文本转换为指定长度的令牌序列。

       参数:
       texts: 待处理的文本，可以是单个字符串或字符串列表。
       context_length: 令牌序列的长度，超过这个长度的序列将被处理。
       truncate: 是否截断超过长度限制的令牌序列。

       返回:
       一个变量张量，其中包含所有文本的分词表示。
    """
    # 检查输入的texts类型，如果是字符串，则转换为列表形式
    if isinstance(texts, str):
        texts = [texts]
    # 获取开始和结束令牌的编号，用于标记序列的开始和结束
    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    # 对每个文本进行编码，并添加开始和结束令牌
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token]
                  for text in texts]

    # 初始化结果张量，用于存放所有令牌序列
    result = jt.zeros((len(all_tokens), context_length), dtype=jt.int64)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            # 如果允许截断，则截断序列并更新结束令牌
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(
                    f"Input {texts[i]} is too long for context length {context_length}"
                )
        # 将处理后的令牌序列填充到结果张量中
        result[i, :len(tokens)] = jt.Var(tokens)

    return result

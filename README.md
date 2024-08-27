# JCLIP

[[Blog]](https://openai.com/blog/clip/) [[Paper]](https://arxiv.org/abs/2103.00020)


JCLIP为CLIP的Jittor版本，CLIP（Contrastive Language-Image Pre-Training）是一个在各种（图像、文本）对上训练的神经网络。可以用自然语言指示它在给定图像的情况下预测最相关的文本片段，而无需直接对任务进行优化，这与 GPT-2和3的zero-shot功能类似。


## 安装依赖环境
```bash
conda create --name jittor python==3.10
conda activate jittor
pip install -r requirements.txt
python setup.py develop
```


## 方法的详细思路

### 项目简介
本项目实现了基于Jittor的Tip-Adapter方法，用于在少样本学习任务中提升CLIP模型的性能。核心思路是通过提取图像和文本特征，构建特征缓存（Cache），并在推理阶段利用该缓存进行特征对比，从而提高模型的分类准确率。

### 详细思路
特征提取与缓存构建：
    图像特征提取：使用预训练的CLIP模型提取训练集和测试集的图像特征，所有图像特征在提取时都会被归一化处理。
    数据增强：对训练集中选择的四张代表性图像进行数据增强，具体增强方法包括随机水平翻转和亮度调整、颜色抖动等。增强后的图像特征将与原始图像特征一起用于训练。
    文本特征提取：为每个类别生成对应的文本特征。文本模板根据类别的不同进行调整，例如“Animal”类别的模板为“a photo of a {}, a type of animal.”，同时支持从JSON文件加载预定义的模板。
    特征缓存：将少量视觉特征和对应的标签作为缓存存储，用于在推理阶段与测试集特征进行对比。

模型推理与预测：
    在推理阶段，通过使用少量视觉特征的缓存模型以及测试集图像特征，计算类别间的相似性得分（affinity），并最终得到预测结果。模型将输出每张测试图像对应的前五个预测类别。


## 预训练模型
CLIP模型: ViT-B-32
https://github.com/openai/CLIP


## 训练过程

### 模型权重

下载[VIT-B-32](https://github.com/uyzhang/JCLIP/releases/tag/%E6%9D%83%E9%87%8D)或利用转换脚本，将PyTorch权重转换为Jittor权重。

并将其放在根目录下

```python
import torch
import jittor as jt
clip = torch.load('ViT-B-32.pt').state_dict()

for k in clip.keys():
    clip[k] = clip[k].float().cpu()
jt.save(clip, 'ViT-B-32.pkl')
```

### 将数据集存放至Dataset文件夹下
包括数据集路径：class_dir(class_b.txt路径)，train_dir(train.txt路径)，imgs_dir(Dataset文件夹路径，里面存放训练集。注意baseline_ft，extract_Feature，predict_ft里都要修改！！！)，train_imgs_dir(predict里路径，也为Dataset路径)

以及保存路径：features_path(初始图像特征保存路径)，权重保存路径
```python
class_dir = 'Dataset/classes_b.txt'
train_dir = 'Dataset/train.txt'
features_path = 'Dataset/img_feature/features.pkl'
imgs_dir = 'Dataset/'
```

### Extracting Features
下载完模型权重和数据集后，整个项目的结构如下:
```
JCLIP/
|–– Dataset
|–––– img_feature
|–––––– features.pkl
|–––– classes.txt
|–––– classes_b.txt
|–––– train.txt
|–––– TrainSet
|–––– TestSetB
|–– jclip
|–––– ... 6 other python files
|–– output
|–– prompts
|–––– Animal_prompt.json
|–––– ... 4 other dataset json files
|–– Augmentation.py
|–– baseline_ft.py
|–– CustomDataset.py
|–– extract_Feature.py
|–– SEAttention.py
|–– setup.py
|–– test.py
|–– Tip_Adapter_utils.py
|–– tools.py
|–– ViT-B-32.pkl
|–– README.md
```



## 第四届计图比赛
- 运行baseline_ft.py得到模型权重
```bash
python baseline_ft.py
```
- 预测结果，运行test.py得到result.txt文件
```bash
python test.py
```

## 联系方式
QQ：1009501558
import jittor as jt
from PIL import Image
import jclip as clip
import os
from tqdm import tqdm
import argparse
import numpy as np
from extract_Feature import extract_img_feature, KM_select_four_img
from CustomDataset import TrainCustomDataset
from Augmentation import apply_data_augmentation, color_jitter, Random_HorizontalFlip_brightness
from tools import save_features, load_features
from jittor.dataset import DataLoader
import shutil
import jittor.nn as nn
from  Tip_Adapter_utils import text_feature, textold_feature, extract_few_shot_feature, extract_val_test_feature, run_tip_adapter, run_tip_adapter_F


jt.flags.use_cuda = 1
jt.set_global_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, default='B')
args = parser.parse_args()

clip_model, preprocess = clip.load("ViT-B-32.pkl")
best_dir = '/root/autodl-tmp/Dataset/output/best_HB_color.pkl'
if not os.path.exists(best_dir):
    print("缺少权重文件，请先运行baseline_ft.py")

checkpoint = jt.load(best_dir)
clip_model.load_state_dict(checkpoint["model"])
print("epoch",checkpoint["epoch"] + 1)

class_dir = 'Dataset/classes_b.txt'
train_dir = 'Dataset/train.txt'
features_path = 'Dataset/img_feature/features.pkl'
imgs_dir = 'Dataset/'

if not os.path.exists(features_path):
    features = extract_img_feature(clip_model, train_dir, imgs_dir, preprocess)  # 用clip提取图像特征,所有图像归一化后
    save_features(features, features_path)
else:
    features = load_features(features_path)

new_train_imgs, new_train_labels, remaining_imgs, remaining_labels = KM_select_four_img(features)


# 根据路径提取图像
print('Training data processing:')
all_four_img_features = []
train_features = []
HB_augmented_features = []
colorjitter_augmented_features = []
vflip_augmented_features = []
with jt.no_grad():
    for img in tqdm(new_train_imgs):
        img = os.path.join(imgs_dir, img)
        image = Image.open(img)
        preprocessed_image = preprocess(image).unsqueeze(0) # [1,3,224,224,]
        train_features.append(preprocessed_image)
        # 数据增强
        HB_image_features = apply_data_augmentation(image,Random_HorizontalFlip_brightness,preprocess)
        HB_augmented_features.append(HB_image_features)
        colorjitter_image_features = apply_data_augmentation(image,color_jitter,preprocess)
        colorjitter_augmented_features.append(colorjitter_image_features)
        # 初始四张图片
        all_four_img_features.append(preprocessed_image)

train_features.extend(HB_augmented_features)
train_features.extend(colorjitter_augmented_features)
train_features = jt.array(jt.concat(train_features, dim=0))
train_labels = jt.array(jt.concat(new_train_labels + new_train_labels + new_train_labels, dim=0))
print("特征训练完成")

all_four_img_features = jt.array(jt.concat(all_four_img_features, dim=0))
all_four_img_labels = jt.array(jt.concat(new_train_labels, dim=0))


# 划分验证集和训练集
val_features = []
with jt.no_grad():
    for img in tqdm(remaining_imgs):
        img = os.path.join(imgs_dir, img)
        image = Image.open(img)
        preprocessed_image = preprocess(image).unsqueeze(0) # [1,3,224,224,]
        val_features.append(preprocessed_image)
val_features = jt.array(jt.concat(val_features, dim=0))
val_labels = jt.array(jt.concat(remaining_labels, dim=0))
print("训练集验证集划分完成")

# -------------------------------------------------------------------

split = 'TestSet' + args.split
imgs_dir = 'Dataset/' + split  # Dataset/TestSetB
test_imgs = os.listdir(imgs_dir)

print('Testing data processing:')

test_features = []
with jt.no_grad():
    for img in tqdm(test_imgs):
        img_path = os.path.join(imgs_dir, img)
        image = Image.open(img_path)
        image = preprocess(image).unsqueeze(0)
        # test_features.append(image)

        # 模型提取特征
        image_features = clip_model.encode_image(image)
        # 对图像特征进行归一化处理
        image_features /= image_features.norm(dim=-1, keepdim=True)
        test_features.append(image_features)
test_features = jt.array(jt.cat(test_features))
print("测试数据特征提取完成")



# 构造数据集
train_dataset = TrainCustomDataset(train_features, train_labels)
val_dataset = TrainCustomDataset(val_features, val_labels)
all_dataset = TrainCustomDataset(all_four_img_features, all_four_img_labels)


train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,num_workers = 0)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True,num_workers = 0)
all_loader = DataLoader(all_dataset, batch_size=16, shuffle=False,num_workers = 0)



# Textual features
print("\nGetting textual features as CLIP's classifier.")
keys = ["Animal", "Caltech-101", "Food-101", "Thu-dog", "Stanford-Cars"]
# 构造文本特征
clip_weights_path = 'Dataset/text_clip_prompts_weights.pkl'
if not os.path.exists(clip_weights_path):
    clip_weights = text_feature(class_dir, clip_model, keys)
    save_features(clip_weights, clip_weights_path)
else:
    clip_weights = load_features(clip_weights_path) # [512,403,]
    print(clip_weights.shape)


print("\nConstructing cache model by few-shot visual features and labels.")
# 构建缓存
cache_keys_path = 'Dataset/Tip_catch_keys.pkl'
cache_values_path = 'Dataset/Tip_catch_values.pkl'
if not os.path.exists(cache_keys_path) and not os.path.exists(cache_values_path):
    cache_keys, cache_values = extract_few_shot_feature(clip_model, all_loader, cache_keys_path, cache_values_path, num_classes=403)
    cache_keys, cache_values = jt.array(cache_keys), jt.array(cache_values).astype(jt.float32)  # [512,364,] [364,1,91,]
    print("缓存构造成功")
else:
    cache_keys = load_features(cache_keys_path)
    cache_values = load_features(cache_values_path)
    cache_keys, cache_values = jt.array(cache_keys), jt.array(cache_values).astype(jt.float32)

# Pre-load val features
print("\nLoading visual features and labels from val set.")
val_features,val_labels = extract_val_test_feature(clip_model, val_loader)  # [73,512,]



# ------------------------------------------------------------------------------------
preds = run_tip_adapter_F(cache_keys, cache_values, val_features, val_labels, test_features, clip_weights, clip_model, train_loader)

# predict
with open('output/result.txt', 'w') as save_file:
    i = 0
    for prediction in preds:
        prediction = np.asarray(prediction) # prediction是个列表，里面是374个种类概率
        top5_idx = prediction.argsort()[-1:-6:-1]
        save_file.write(test_imgs[i] + ' ' +
                        ' '.join(str(idx) for idx in top5_idx) + '\n')
        i += 1
print("写入成功")




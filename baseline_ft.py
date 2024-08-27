import jittor as jt
from PIL import Image
import jclip as clip
import os
from tqdm import tqdm
import jittor as jt
from PIL import Image
import jclip as clip
import os
from tqdm import tqdm
import argparse
import numpy as np
from extract_Feature import extract_img_feature, KM_select_four_img, contrastive_loss_plus, text_feature
from CustomDataset import TrainCustomDataset, CustomCLIP
from Augmentation import Random_HorizontalFlip_brightness, apply_data_augmentation, color_jitter
from tools import save_features, load_features
from jittor.dataset import DataLoader
import shutil

jt.flags.use_cuda = 1

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, default='A')
args = parser.parse_args()

model, preprocess = clip.load("ViT-B-32.pkl")
# class_dir = 'Dataset/classes.txt'
class_dir = 'Dataset/classes_b.txt'
train_dir = 'Dataset/train.txt'
features_path = 'Dataset/img_feature/features.pkl'
imgs_dir = 'Dataset/'

if not os.path.exists(features_path):
    features = extract_img_feature(model, train_dir, imgs_dir, preprocess)  # 用clip提取图像特征,所有图像归一化后
    save_features(features, features_path)
else:
    features = load_features(features_path)

new_train_imgs, new_train_labels, remaining_imgs, remaining_labels = KM_select_four_img(features)  # 1496

print("\nGetting textual features as CLIP's classifier.")
keys = ["Animal", "Caltech-101", "Food-101", "Thu-dog", "Stanford-Cars"]
# 构造文本特征
clip_weights_path = 'Dataset/text_clip_prompts_weights.pkl'
if not os.path.exists(clip_weights_path):
    clip_weights = text_feature(class_dir, model, keys)
    save_features(clip_weights, clip_weights_path)
else:
    clip_weights = load_features(clip_weights_path)  # [512,403,]
    print(clip_weights.shape)

train_features = []
HB_augmented_features = []
colorjitter_augmented_features = []
print('Training data processing:')
with jt.no_grad():
    for img in tqdm(new_train_imgs):
        img = os.path.join(imgs_dir, img)
        image = Image.open(img)
        preprocessed_image = preprocess(image).unsqueeze(0)  # [1,3,224,224,]
        train_features.append(preprocessed_image)
        # 数据增强
        HB_image_features = apply_data_augmentation(image, Random_HorizontalFlip_brightness, preprocess)
        HB_augmented_features.append(HB_image_features)
        colorjitter_image_features = apply_data_augmentation(image, color_jitter, preprocess)
        colorjitter_augmented_features.append(colorjitter_image_features)
print("特征训练完成")

train_features.extend(HB_augmented_features)
train_features.extend(colorjitter_augmented_features)  # 合并特征列表
train_features = jt.array(jt.concat(train_features, dim=0))  # [2992,3,224,224,]
train_labels = jt.array(jt.concat(new_train_labels + new_train_labels + new_train_labels, dim=0))  # [2992,]
aug_text_features = jt.array(clip_weights)

print('Validating data processing:')
val_features = []
with jt.no_grad():
    for img in tqdm(remaining_imgs):
        img = os.path.join(imgs_dir, img)
        image = Image.open(img)
        preprocessed_image = preprocess(image).unsqueeze(0)  # [1,3,224,224,]
        val_features.append(preprocessed_image)
val_features = jt.array(jt.concat(val_features, dim=0))
val_labels = jt.array(jt.concat(remaining_labels, dim=0))

# 构造数据集
train_dataset = TrainCustomDataset(train_features, train_labels)
val_dataset = TrainCustomDataset(val_features, val_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=0)
print("数据集构造完成")

clip_model = CustomCLIP(model)
criterion = jt.nn.CrossEntropyLoss()

optimizer = jt.optim.SGD(clip_model.parameters(), lr=0.001, momentum=0.9,
                         weight_decay=1e-4)  # weight_decay：1e-4 2.22 lr：1e-2 0.0001
lr_scheduler = jt.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 25], gamma=0.5)

print('Training data processing:')
num_epochs = 35
print_every = 1
best_acc = 0.0
contrastive_loss_weight = 1.0
for epoch in range(num_epochs):
    clip_model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc="Training"):
        # 前向传播
        logits_per_image, logits_per_text = clip_model(inputs, aug_text_features)
        loss = criterion(logits_per_image, labels)
        optimizer.zero_grad()
        optimizer.backward(loss)
        optimizer.clip_grad_norm(0.1, 2)
        optimizer.step()

        running_loss += loss.data[0]
    lr_scheduler.step()

    # 打印训练损失
    if (epoch + 1) % print_every == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

    # 验证循环
    clip_model.eval()
    with jt.no_grad():
        total_correct = 0
        for inputs, labels in tqdm(val_loader, desc="Validation"):
            logits_per_image, logits_per_text = clip_model(inputs, aug_text_features)  # [8,374,]
            max_indices, max_probs = jt.argmax(logits_per_image, 1)  # [8,]
            labels = jt.flatten(labels)
            correct_count = (max_indices == labels).sum().item()
            total_correct += correct_count

        acc_valid = total_correct / len(val_loader.dataset)
        print(f"Validation Accuracy: {acc_valid * 100:.3f}%")

    checkpoint = {
        "model": clip_model.state_dict(),
        "epoch": epoch,
    }
    jt.save(checkpoint, f"output/last_HB_color.pkl")

    if best_acc < acc_valid:
        # 保存验证集上表现最好的模型
        best_acc, best_epoch = acc_valid, epoch
        shutil.copy(f"output/last_HB_color.pkl",
                    f"output/best_HB_color.pkl")

print(f"best_acc: {best_acc}")
print(f"best_epoch: {best_epoch}")

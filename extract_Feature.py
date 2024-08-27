import jittor as jt
import jclip as clip
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import pdb
from typing import Dict
import jittor.nn as nn
import random
import json


def contrastive_loss_plus(features, labels, temperature=0.5):
    features_flat = features / jt.sqrt((features ** 2).sum(1, keepdims=True))
    features_flat_np = features_flat.view(features.size(0), -1).numpy()
    similarity_matrix = cosine_similarity(features_flat_np)  # (16, 16)
    similarity_matrix = jt.array(similarity_matrix)

    loss = 0.0
    num_features = len(features)  # 16

    unique_labels = np.unique(labels.numpy())  # 获取唯一类别标签

    for label in unique_labels:
        pos_indices = np.where(labels.numpy() == label)[0]
        neg_indices = np.where(labels.numpy() != label)[0]

        for i in pos_indices:
            pos_similarities = similarity_matrix[i, pos_indices].tolist()
            pos_similarities.pop(pos_indices.tolist().index(i))  # 去除本身的相似度
            if len(pos_similarities) == 0:
                continue
            pos_similarities = jt.array(pos_similarities)  # []

            neg_similarities = similarity_matrix[i, neg_indices]

            pos_term = jt.nn.logsumexp(pos_similarities / temperature, dim=0)

            k = min(neg_similarities.shape[0], 4)  # 这里选取前k-1个负样本，k可以调整
            top_neg_similarities = jt.topk(neg_similarities, k)[0]
            neg_term = jt.nn.logsumexp(top_neg_similarities / temperature, dim=0)

            loss += -(pos_term - neg_term)

    return loss / num_features


def extract_representative_features_by_mean(model, dataloader, num_classes=374):
    features = []
    labels = []

    # 提取所有样本的特征和标签
    with jt.no_grad():
        for inputs, label in dataloader:
            features.append(model.encode_image(inputs))
            labels.append(label)

    features = jt.concat(features)
    labels = jt.concat(labels)

    # 按类别分组特征
    unique_labels = np.unique(labels.numpy())
    representative_features = []
    representative_labels = jt.array(unique_labels).astype(jt.int32)
    representative_labels = jt.nn.one_hot(representative_labels, num_classes=num_classes).half()
    for label in unique_labels:
        idx = (labels == label).nonzero()
        idx_first_column = idx[:, 0]
        class_features = features[idx_first_column]  # [4,512,]

        # 计算每个类别特征的平均值
        mean_features = class_features.mean(dim=0, keepdim=True)
        mean_features /= mean_features.norm(dim=-1, keepdim=True)
        representative_features.append(mean_features)

    return representative_features, representative_labels


def random_select_four_img(train_dir):
    train_labels = open(train_dir).read().splitlines()
    train_imgs_path = [l.split(' ')[0] for l in train_labels]
    train_imgs_labels = [jt.float32([int(l.split(' ')[1])]) for l in train_labels]

    # 筛选每个类别的前四张图像
    imgs_dir = '/root/autodl-tmp/Dataset/'
    cnt = {}
    new_train_imgs = []
    new_train_labels = []
    for i in range(len(train_imgs_path)):
        label = int(train_imgs_labels[i].numpy())
        # 如果当前标签不在计数字典中，则初始化其计数为0
        if label not in cnt:
            cnt[label] = 0
        if cnt[label] < 4:
            img_path = os.path.join(imgs_dir, train_imgs_path[i])
            new_train_imgs.append(img_path)  # 此处是图片路径
            new_train_labels.append(train_imgs_labels[i])
            cnt[label] += 1
    return new_train_imgs, new_train_labels


def DB_select_four_img(features):
    new_train_imgs = []
    new_train_labels = []
    required_images_per_label = 4

    for label in features:
        data = features[label]
        # 提取图像路径和特征
        img_paths, img_features = zip(*data)
        # 获取所有唯一的聚类标签
        img_features = np.vstack(img_features)

        dbscan = DBSCAN(eps=0.5, min_samples=2).fit(img_features)
        # 获取聚类标签
        labels = dbscan.labels_
        unique_labels = set(labels)
        representative_images = []
        for cluster_label in unique_labels:
            # 跳过噪声点，即标签为-1的点
            if cluster_label == -1:
                continue
            cluster_indices = np.where(labels == cluster_label)[0]
            cluster_features = img_features[cluster_indices]
            # 计算当前聚类的质心
            cluster_centroid = np.mean(cluster_features, axis=0)
            # 找到距离质心最近的图像索引
            closest_idx = cluster_indices[np.argmin(np.linalg.norm(cluster_features - cluster_centroid, axis=1))]
            representative_images.append((img_paths[closest_idx], label))
        # 如果代表性图像少于4张，从较大的簇中选择多个代表性图像
        if len(representative_images) < required_images_per_label:
            for cluster_label in unique_labels:
                if cluster_label == -1:
                    continue
                cluster_indices = np.where(labels == cluster_label)[0]
                cluster_features = img_features[cluster_indices]
                cluster_centroid = np.mean(cluster_features, axis=0)

                for idx in cluster_indices:
                    if len(representative_images) >= required_images_per_label:
                        break
                    closest_idx = idx
                    if (img_paths[closest_idx], label) not in representative_images:
                        representative_images.append((img_paths[closest_idx], label))
        else:
            representative_images = representative_images[:required_images_per_label]

        for img_path, label in representative_images:
            new_train_imgs.append(img_path)
            new_train_labels.append(jt.float32([label]))
    return new_train_imgs, new_train_labels


def KM_select_four_img(features, remain_num=4):
    new_train_imgs = []
    new_train_labels = []
    remaining_imgs = []
    remaining_labels = []
    for label in features:
        # 获取图像特征
        data = features[label]
        # 提取图像路径和特征
        img_paths, img_features = zip(*data)
        # 将图像特征堆叠成一个矩阵，方便进行聚类分析
        img_features = np.vstack(img_features)

        kmeans = KMeans(n_clusters=4, random_state=0).fit(img_features)
        # 获取聚类中心点
        centers = kmeans.cluster_centers_

        # 遍历每个聚类中心，找到与其最近的图像，并将其路径和标签添加到新的训练数据集中
        num_images_per_label = 0  # 用于统计当前类别的图片数量
        selected_indices = []
        for center in centers:
            cluster_indices = np.where(kmeans.labels_ == num_images_per_label)[0]
            cluster_features = img_features[cluster_indices]
            cluster_centroid = np.mean(cluster_features, axis=0)
            closest_idx = cluster_indices[np.argmin(np.linalg.norm(cluster_features - cluster_centroid, axis=1))]

            new_train_imgs.append(img_paths[closest_idx])
            new_train_labels.append(jt.float32([label]))
            selected_indices.append(closest_idx)

            num_images_per_label += 1  # 增加当前类别的图片数量
            if num_images_per_label >= 4:  # 控制每个类别只选择四张图片
                break
        remaining_indices = [idx for idx in range(remain_num) if idx not in selected_indices]
        remaining_imgs.extend([img_paths[idx] for idx in remaining_indices])
        remaining_labels.extend([jt.float32([label]) for _ in range(len(remaining_indices))])
    return new_train_imgs, new_train_labels, remaining_imgs, remaining_labels


def select_representative_images(image_features, model, top_k=4, remain_num=4):
    new_train_imgs, new_train_labels, remaining_imgs, remaining_labels = [], [], [], []
    with jt.no_grad():
        for label in image_features:
            # 获取图像特征
            data = image_features[label]
            # 提取图像路径和特征
            img_paths, img_features = zip(*data)
            # 将图像特征堆叠成一个矩阵，方便进行聚类分析
            img_features = jt.array(np.vstack(img_features))  # [84,512]
            classname = img_paths[0].split('/')[2]
            # TODO:更具体点
            text = jt.cat([clip.tokenize(f"a photo of a {classname}")])
            text_features = model.encode_text(text).permute(1, 0)  # [512,1,]
            # 计算图像特征和文本特征的相似度
            similarity = (100.0 * img_features @ text_features)  # [84,1,]
            combined = zip(img_paths, similarity)
            sorted_combinations = sorted(combined, key=lambda x: x[1], reverse=True)
            top_4_img_paths, top_4_similarity = zip(*sorted_combinations[:top_k])
            new_train_imgs.extend(top_4_img_paths)
            new_train_labels.extend([jt.array(label)] * top_k)

            # 从剩余图像中随机选择 4 张图像
            remaining_img_paths, remaining_img_features = zip(*sorted_combinations[top_k:])
            if len(remaining_img_paths) > 0:
                num_random_imgs = min(remain_num, len(remaining_img_paths))
                remaining_random_img_paths = random.sample(remaining_img_paths, num_random_imgs)
                remaining_imgs.extend(remaining_random_img_paths)
                remaining_labels.extend([jt.array(label)] * num_random_imgs)
            else:
                continue

    return new_train_imgs, new_train_labels, remaining_imgs, remaining_labels


def KM_select_random_one_img(new_train_imgs, new_train_labels):
    # combined_data = zip(new_train_imgs, new_train_labels)
    class_indices = {}

    # 创建一个字典来存储每个类别的索引
    for idx, label in enumerate(new_train_labels):
        label_value = int(label.item())
        if label_value not in class_indices:
            class_indices[label_value] = []
        class_indices[label_value].append(idx)

    # 从每个类别中随机选择一张图片
    rep_train_imgs = []
    rep_train_labels = []
    for label in class_indices:
        img_idx = random.choice(class_indices[label])
        rep_train_imgs.append(new_train_imgs[img_idx])
        rep_train_labels.append(label)

    return rep_train_imgs, rep_train_labels


def KM_select_two_img(features):
    closest_to_center_imgs = []  # 新增列表，用于存储距离聚类中心最近的图片路径
    closest_to_center_labels = []  # 新增列表，用于存储距离聚类中心最近的图片标签

    for label in features:
        # 获取图像特征
        data = features[label]
        # 提取图像路径和特征
        img_paths, img_features = zip(*data)
        # 将图像特征堆叠成一个矩阵，方便进行聚类分析
        img_features = np.vstack(img_features)

        kmeans = KMeans(n_clusters=4, random_state=0).fit(img_features)
        # 获取聚类中心点
        centers = kmeans.cluster_centers_

        # 遍历每个聚类中心，找到与其最近的图像，并将其路径和标签添加到新的训练数据集中
        num_images_per_label = 0  # 用于统计当前类别的图片数量
        for center in centers:
            cluster_indices = np.where(kmeans.labels_ == num_images_per_label)[0]
            cluster_features = img_features[cluster_indices]
            cluster_centroid = np.mean(cluster_features, axis=0)
            closest_idx = cluster_indices[np.argmin(np.linalg.norm(cluster_features - cluster_centroid, axis=1))]

            closest_to_center_imgs.append(img_paths[closest_idx])  # 将距离聚类中心最近的图片路径添加到列表中
            closest_to_center_labels.append(jt.float32([label]))  # 将距离聚类中心最近的图片标签添加到列表中

            num_images_per_label += 1  # 增加当前类别的图片数量
            if num_images_per_label >= 2:  # 控制每个类别只选择四张图片
                break
    return closest_to_center_imgs, closest_to_center_labels


def textold_feature(class_dir):
    classes = open(class_dir).read().splitlines()
    num_classes = len(classes)
    new_classes = []
    for c in classes:  # 374
        c = c.split(' ')[0]
        if c.startswith('Animal'):
            c = c[7:]
            template = 'a photo of a {}, a type of animal.'
            template = template.format(c)
        elif c.startswith('Thu-dog'):
            c = c[8:]
            template = 'a photo of a {}, a type of dog.'
            template = template.format(c)
        elif c.startswith('Caltech-101'):
            c = c[12:]
            template = 'a photo of a ' + c
        elif c.startswith('Food-101'):
            c = c[9:]
            template = 'a photo of a {}, a type of food.'
            template = template.format(c)
        elif c.startswith('Stanford-Cars'):
            c = c[14:]
            template = 'a photo of a {}, a type of car.'
            template = template.format(c)
        elif c.startswith('Stanford-Cars'):
            c = c[14:]
            template = 'a photo of a {}, a type of car.'
            template = template.format(c)
        new_classes.append(template)
    text = clip.tokenize(new_classes)  # [403,77,]
    return num_classes, text


def text_class_feature(class_dir):
    '''
    将各自数据集的种类添加到各自的列表中
    '''
    classes = open(class_dir).read().splitlines()
    num_classes = len(classes)
    new_classes = {}
    Animal_classes = []
    Thu_dog_classes = []
    Caltech101_classes = []
    Food101_classes = []
    StanfordCars_classes = []
    for c in classes:  # 374
        c = c.split(' ')[0]
        if c.startswith('Animal'):
            c = c[7:]
            Animal_classes.append(c)
        elif c.startswith('Thu-dog'):
            c = c[8:]
            Thu_dog_classes.append(c)
        elif c.startswith('Caltech-101'):
            c = c[12:]
            Caltech101_classes.append(c)
        elif c.startswith('Food-101'):
            c = c[9:]
            Food101_classes.append(c)
        elif c.startswith('Stanford-Cars'):
            c = c[14:]
            StanfordCars_classes.append(c)
    new_classes["Animal"] = Animal_classes
    new_classes["Thu-dog"] = Thu_dog_classes
    new_classes["Caltech-101"] = Caltech101_classes
    new_classes["Food-101"] = Food101_classes
    new_classes["Stanford-Cars"] = StanfordCars_classes
    return new_classes


def extract_text_feature(classnames, clip_model, template, prompt_path=None):
    clip_weights = []
    if prompt_path:
        f = open(prompt_path)
        prompts = json.load(f)
        with jt.no_grad():
            for classname in classnames:
                classname = classname.replace('_', ' ')
                template_texts = template.format(classname)
                textslist = []
                textslist.append(template_texts)
                if classname in prompts:
                    for t in prompts[classname]:
                        textslist.append(t)
                texts_token = clip.tokenize(textslist)
                class_embeddings = clip_model.encode_text(texts_token)
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)  # [51,512,]
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()  # [512,]
                clip_weights.append(class_embedding)

            clip_weights = jt.stack(clip_weights, dim=1)
    else:
        print("缺少json文件")
        # with jt.no_grad():
        #     for classname in classnames:
        #         classname = classname.replace('_', ' ')
        #         texts = [template.format(classname)]
        #         texts_token = clip.tokenize(texts, truncate=True)
        #         class_embeddings = clip_model.encode_text(texts_token)
        #         class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)

        #         class_embedding = class_embeddings.mean(dim=0)
        #         class_embedding /= class_embedding.norm()
        #         clip_weights.append(class_embedding)
        #     clip_weights = jt.stack(clip_weights, dim=1)  # [feature_dim,feature_num,]
    return clip_weights


def text_feature(class_dir, clip_model, keys):
    classdict = text_class_feature(class_dir)
    clip_weights = []
    for key in keys:
        classnames = classdict[key]
        if key == "Animal":
            template = 'a photo of a {}, a type of animal.'
            prompt_path = 'prompts/Animal_prompt.json'
            if not os.path.exists(prompt_path):
                print("没有找到Animal_prompt.json文件,提取文本特征失败")
            text_weights = extract_text_feature(classnames, clip_model, template, prompt_path)
        elif key == "Thu-dog":
            template = 'a photo of a {}, a type of dog.'
            prompt_path = 'prompts/thudog_prompt.json'
            if not os.path.exists(prompt_path):
                print("没有找到thudog_prompt.json文件,提取文本特征失败")
            text_weights = extract_text_feature(classnames, clip_model, template, prompt_path)
        elif key == "Caltech-101":
            template = 'a photo of a {}.'
            prompt_path = 'prompts/caltech_prompt.json'
            if not os.path.exists(prompt_path):
                print("没有找到caltech_prompt.json文件,提取文本特征失败")
            text_weights = extract_text_feature(classnames, clip_model, template, prompt_path)  # [512,91,]
        elif key == "Food-101":
            template = 'a photo of a {}, a type of food.'
            prompt_path = 'prompts/food101_prompt.json'
            if not os.path.exists(prompt_path):
                print("没有找到food101_prompt.json文件,提取文本特征失败")
            text_weights = extract_text_feature(classnames, clip_model, template, prompt_path)  # [512,101,]
        elif key == "Stanford-Cars":
            template = 'a photo of a {}, a type of car.'
            prompt_path = 'prompts/stanfordcars_prompt.json'
            if not os.path.exists(prompt_path):
                print("没有找到stanfordcars_prompt.json文件,提取文本特征失败")
            text_weights = extract_text_feature(classnames, clip_model, template, prompt_path)
        clip_weights.append(text_weights)
    clip_weights = jt.cat(clip_weights, dim=1)  # [512,403,]
    return clip_weights


def extract_train_img_feature(model, train_dir, transform=None):
    new_train_imgs, new_train_labels = random_select_four_img(train_dir)
    # 初始化一个空列表，用于存储从训练图像中提取的特征
    train_features = []
    print('img_feature extracting:')
    with jt.no_grad():
        for img in tqdm(new_train_imgs):
            image = Image.open(img)
            if transform:
                image = transform(image).unsqueeze(0)
            image_features = model.encode_image(image)
            # 对图像特征进行归一化处理
            image_features /= image_features.norm(dim=-1, keepdim=True)
            # 将处理后的特征添加到训练特征列表中
            train_features.append(image_features)
    train_features = jt.stack(train_features)
    train_features = train_features.squeeze(1)
    return train_features


def extract_img_feature(model, train_dir, imgs_dir, transform=None):
    train_labels = open(train_dir).read().splitlines()
    train_imgs = [l.split(' ')[0] for l in train_labels]
    # train_labels = [jt.float32([int(l.split(' ')[1])]) for l in train_labels]
    train_labels = [int(l.split(' ')[1]) for l in train_labels]  # 将标签转换为整数

    features = {}
    with jt.no_grad():
        for i, img_path in enumerate(tqdm(train_imgs)):
            label = train_labels[i]
            img = os.path.join(imgs_dir, img_path)
            image = Image.open(img)
            preprocessed_image = transform(image).unsqueeze(0)
            image_features = model.encode_image(preprocessed_image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            if label not in features:
                features[label] = []
            features[label].append((img_path, image_features.numpy()))  # 字典存放是图像路径(不包含Dataset)和图像特征
    return features


def extract_img_feature_four(model, train_dir, transform=None):
    imgs_dir = '/root/autodl-tmp/Dataset/'
    train_labels = open(train_dir).read().splitlines()
    train_imgs = [l.split(' ')[0] for l in train_labels]
    # train_labels = [jt.float32([int(l.split(' ')[1])]) for l in train_labels]
    train_labels = [int(l.split(' ')[1]) for l in train_labels]  # 将标签转换为整数

    features = {}
    for i, img_path in enumerate(tqdm(train_imgs)):
        label = train_labels[i]
        img = os.path.join(imgs_dir, img_path)
        image = Image.open(img)
        preprocessed_image = transform(image).unsqueeze(0)
        # image_features = model.encode_image(preprocessed_image)
        # image_features /= image_features.norm(dim=-1, keepdim=True)

        if label not in features:
            features[label] = []
        features[label].append((img_path, preprocessed_image.numpy()))  # 字典存放是图像路径(不包含Dataset)和图像特征
    return features







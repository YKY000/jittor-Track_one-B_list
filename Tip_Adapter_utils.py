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
import pdb
import json
from jittor.dataset import DataLoader
import shutil
import jittor.nn as nn
from extract_Feature import text_class_feature
from SEAttention import SEAttention


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


def textold_feature(class_dir, clip_model):
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
        new_classes.append(template)
    text = clip.tokenize(new_classes)
    class_embeddings = clip_model.encode_text(text)
    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
    return num_classes, class_embeddings


def extract_few_shot_feature(clip_model, train_loader_cache, cache_keys_path, cache_values_path, num_classes,
                             augment_epoch=10):
    # 将整个训练集作为缓存模型
    cache_keys = []
    cache_values = []
    with jt.no_grad():
        # Data augmentation for the cache model
        for augment_idx in range(augment_epoch):
            train_features = []
            print('Augment Epoch: {:} / {:}'.format(augment_idx, augment_epoch))
            for i, (images, target) in enumerate(tqdm(train_loader_cache)):

                image_features = clip_model.encode_image(images)
                train_features.append(image_features)
                if augment_idx == 0:
                    cache_values.append(target)
            # 将当前增强周期的所有特征合并为一个张量,并添加到缓存键的列表中
            cache_keys.append(jt.cat(train_features, dim=0).unsqueeze(0))  # [1,364,512,]

    # 计算所有增强周期特征的平均值,作为最终的缓存键
    cache_keys = jt.cat(cache_keys, dim=0).mean(dim=0)
    cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
    cache_keys = cache_keys.permute(1, 0)  # [512,364,]
    # 将所有目标转换为one-hot编码,并将其半量化,以减小存储空间
    cache_values = jt.nn.one_hot(jt.cat(cache_values, dim=0).astype(jt.int32),
                                 num_classes=num_classes).half()  # [364,1,91,]
    # 将缓存键和值保存到指定的目录中
    jt.save(cache_keys, cache_keys_path)
    jt.save(cache_values, cache_values_path)
    return cache_keys, cache_values


def extract_few_shot_mean_feature(clip_model, train_loader_cache, cache_keys_path, cache_values_path, num_classes,
                                  augment_epoch=10):
    # 将每4张图像的平均值作为缓存值
    cache_keys = []
    cache_values = []
    with jt.no_grad():
        # Data augmentation for the cache model
        for augment_idx in range(augment_epoch):
            train_features = []
            labels = []
            print('Augment Epoch: {:} / {:}'.format(augment_idx, augment_epoch))
            for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                image_features = clip_model.encode_image(images)
                train_features.append(image_features)
                labels.append(target)

            # 按类别分组特征
            train_features = jt.concat(train_features)  # [364,512,]
            labels = jt.concat(labels)  # [364,1,]
            unique_labels = np.unique(labels.numpy())  # (91,)
            if augment_idx == 0:
                cache_values = jt.array(unique_labels).astype(jt.int32)  # [91,]
            train_mean_features = []

            for label in unique_labels:
                idx = (labels == label).nonzero()
                idx_first_column = idx[:, 0]
                class_features = train_features[idx_first_column]  # [4,512,]

                # 计算每个类别特征的平均值
                mean_features = class_features.mean(dim=0, keepdim=True)
                # mean_features /= mean_features.norm(dim=-1, keepdim=True) # [1,512,]
                train_mean_features.append(mean_features)
            # 将当前增强周期的所有特征合并为一个张量,并添加到缓存键的列表中
            cache_keys.append(jt.cat(train_mean_features, dim=0).unsqueeze(0))  # [1,91,512]

    # 计算所有增强周期特征的平均值,作为最终的缓存键
    cache_keys = jt.cat(cache_keys, dim=0).mean(dim=0)  # [91,512,]
    cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
    cache_keys = cache_keys.permute(1, 0)
    # 将所有目标转换为one-hot编码,并将其半量化,以减小存储空间
    cache_values = jt.nn.one_hot(cache_values, num_classes=num_classes).unsqueeze(1).half()  # [91,1,91,]
    # 将缓存键和值保存到指定的目录中
    jt.save(cache_keys, cache_keys_path)
    jt.save(cache_values, cache_values_path)
    return cache_keys, cache_values


def extract_val_test_feature(clip_model, loader):
    # 提取图像特征
    features, labels = [], []
    with jt.no_grad():
        for i, (images, target) in enumerate(tqdm(loader)):
            image_features = clip_model.encode_image(images)
            image_features = SE_Attention(image_features)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            features.append(image_features)
            labels.append(target)
    features, labels = jt.cat(features), jt.cat(labels)
    return features, labels


def SE_Attention(images):
    images = images.unsqueeze(2).unsqueeze(3)
    se = SEAttention(channel=512, reduction=8)
    output = se(images)
    output = output.squeeze(3).squeeze(2)
    return output


# ------------------------------------------ Tip-Adapter ------------------------------------------
def search_hp(cache_keys, cache_values, features, labels, clip_weights, adapter=None, mlp_adapter=None):
    search_scale = [20, 10]
    search_step = [300, 40]
    beta_list = np.logspace(np.log10(0.1), np.log10(search_scale[0]), search_step[0])
    alpha_list = np.logspace(np.log10(0.1), np.log10(search_scale[1]), search_step[1])

    best_acc = 0
    best_beta, best_alpha = 0, 0
    # 遍历所有候选的beta和alpha值
    for beta in tqdm(beta_list, desc="Beta Loop"):
        for alpha in alpha_list:
            if adapter:
                affinity = adapter(features)
            else:
                affinity = features @ cache_keys

            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            clip_logits = 100. * features @ clip_weights
            tip_logits = clip_logits + cache_logits * alpha
            if mlp_adapter:
                mlp_affinity = mlp_adapter(features)
                tip_logits = tip_logits + mlp_affinity * alpha
                acc = cls_acc(tip_logits, labels)
            else:
                acc = cls_acc(tip_logits, labels)

            if acc > best_acc:
                print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                best_acc = acc
                best_beta = beta
                best_alpha = alpha
    print("\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))
    return best_beta, best_alpha


def search_hp_mlp(cache_keys, cache_values, features, labels, clip_weights, adapter=None, mlp_adapter=None):
    search_scale = [20, 10, 5]
    search_step = [300, 40, 20]

    beta_list = np.logspace(np.log10(0.1), np.log10(search_scale[0]), search_step[0])
    alpha_list = np.logspace(np.log10(0.1), np.log10(search_scale[1]), search_step[1])
    gamma_list = np.logspace(np.log10(0.1), np.log10(search_scale[2]), search_step[2])

    best_acc = 0
    best_beta, best_alpha, best_gamma = 0, 0, 0
    # 遍历所有候选的beta和alpha值
    for beta in tqdm(beta_list, desc="Beta Loop"):
        for alpha in alpha_list:
            if adapter:
                affinity = adapter(features)
            else:
                affinity = features @ cache_keys

            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            clip_logits = 100. * features @ clip_weights
            tip_logits = clip_logits + cache_logits * alpha
            if mlp_adapter:
                mlp_affinity = mlp_adapter(features)
                for gamma in gamma_list:
                    tip_logits_mlp = tip_logits + mlp_affinity * gamma
                    acc = cls_acc(tip_logits_mlp, labels)
                    if acc > best_acc:
                        print("New best setting, beta: {:.2f}, alpha: {:.2f}, gamma: {:.2f}; accuracy: {:.2f}".format(
                            beta, alpha, gamma, acc))
                        best_acc = acc
                        best_beta = beta
                        best_alpha = alpha
                        best_gamma = gamma
            else:
                acc = cls_acc(tip_logits, labels)
                if acc > best_acc:
                    print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha

    print("\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))
    return best_beta, best_alpha, best_gamma


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()  # [1,73,]
    correct = (pred == target.view(1, -1).expand_as(pred))  # [1,73,]
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def run_tip_adapter(cache_keys, cache_values, val_features, val_labels, clip_weights, test_features=None):
    print("\n-------- Searching hyperparameters on the val set. --------")

    # 计算零样本CLIP的logits
    clip_logits = 100. * val_features @ clip_weights
    acc = cls_acc(clip_logits, val_labels)
    print("\n**** Zero-shot CLIP's val accuracy: {:.2f}. ****\n".format(acc))

    # 初始化Tip-Adapter的超参数
    # TODO：这里写死了
    beta, alpha = 4, 6
    affinity = val_features @ cache_keys  # 验证集和缓存键图像间的相似度 [73,512,]@[512,364,] -> [73,364,]
    cache_values = cache_values.squeeze(1).astype(jt.float32)  # [364,91,]
    affinity = affinity.astype(jt.float32)

    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values  # [73,91,]
    tip_logits = clip_logits + cache_logits * alpha
    acc = cls_acc(tip_logits, val_labels)
    print("**** Tip-Adapter's val accuracy: {:.2f}. ****\n".format(acc))

    # 搜索最佳的超参数beta和alpha
    best_beta, best_alpha = search_hp(cache_keys, cache_values, val_features, val_labels, clip_weights)
    print("best_beta:", best_beta)
    print("best_alpha:", best_alpha)

    print("\n-------- Evaluating on the test set. --------")
    # 计算零样本CLIP在测试集上的logits和准确率
    # clip_logits = 100. * test_features @ clip_weights
    # acc = cls_acc(clip_logits, test_labels)
    # print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(acc))

    # 使用最佳超参数计算Tip-Adapter在测试集上的logits和准确率
    affinity = val_features @ cache_keys
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values

    tip_logits = clip_logits + cache_logits * best_alpha
    acc = cls_acc(tip_logits, val_labels)
    print("**** Tip-Adapter's test accuracy: {:.2f}. ****\n".format(acc))


def run_tip_adapter_F(cache_keys, cache_values, val_features, val_labels, test_features, clip_weights, clip_model,
                      train_loader_F, selected_val_features=None, selected_val_labels=None):
    # 初始化adapter层，用于学习缓存键和特征之间的关系
    adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False)  # 512->364
    adapter.weight.data = adapter.weight.data.astype(str(clip_model.dtype))
    adapter.weight = nn.Parameter(cache_keys.t())

    # 新特征
    adapter_mlp = nn.Sequential(nn.Linear(cache_keys.shape[0], 1024, bias=False),
                                nn.BatchNorm1d(1024),
                                # nn.ReLU(),
                                nn.LeakyReLU(0.05),
                                nn.Linear(1024, 403, bias=False))

    optimizer = jt.optim.AdamW(adapter.parameters() + adapter_mlp.parameters(), lr=0.001, eps=1e-4)
    scheduler = jt.lr_scheduler.CosineAnnealingLR(optimizer, 60 * len(train_loader_F))

    beta, alpha = 1, 1.17
    best_acc, best_epoch = 0.0, 0
    cache_values = cache_values.squeeze(1).astype(jt.float32)  # [364,91,]
    for train_idx in range(60):
        # Train
        adapter.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, 60))

        for i, (images, target) in enumerate(tqdm(train_loader_F)):
            with jt.no_grad():
                image_features = clip_model.encode_image(images)  # [16,512]
                image_features = SE_Attention(image_features)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            affinity = adapter(image_features)
            affinity = affinity.astype(jt.float32)

            affinity_mlp = adapter_mlp(image_features)
            affinity_mlp = affinity_mlp.astype(jt.float32)

            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            clip_logits = 100. * image_features @ clip_weights
            tip_logits = clip_logits + cache_logits * alpha

            loss = jt.nn.cross_entropy_loss(tip_logits, target)
            l1_loss = jt.nn.l1_loss(affinity_mlp, clip_logits)  # [16,403,] [16,403,]
            mlp_loss = jt.nn.cross_entropy_loss(affinity_mlp, target)
            loss = l1_loss + loss + mlp_loss
            tip_logits = tip_logits + affinity_mlp * alpha

            acc = cls_acc(tip_logits, target)
            correct_samples += acc / 100 * len(tip_logits)
            all_samples += len(tip_logits)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            optimizer.backward(loss)
            optimizer.step()
            scheduler.step()

        # current_lr = scheduler.get_last_lr()[0]
        print('Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(correct_samples / all_samples, int(correct_samples),
                                                           all_samples, sum(loss_list) / len(loss_list)))

        # Eval
        adapter.eval()
        # 在测试集上进行评估
        affinity = adapter(val_features).astype(jt.float32)
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        clip_logits = 100. * val_features @ clip_weights
        tip_logits = clip_logits + cache_logits * alpha

        affinity_mlp = adapter_mlp(val_features).astype(jt.float32)
        tip_logits = tip_logits + affinity_mlp * alpha

        acc = cls_acc(tip_logits, val_labels)

        # 更新最佳精度和对应的epoch
        print("**** Tip-Adapter-F's val accuracy: {:.2f}. ****\n".format(acc))
        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx + 1
            jt.save(adapter.weight, f"Dataset/best_F_mlp_Tip.pkl")

    adapter.weight = jt.array(jt.load('Dataset/best_F_mlp_Tip.pkl'))
    print(f"**** After fine-tuning, Tip-Adapter-F's best val accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")

    print("\n-------- Searching hyperparameters on the val set. --------")

    # 在验证集上搜索超参数
    best_beta, best_alpha = 2.17, 0.23
    # 在测试集上进行最终评估
    print("\n-------- Evaluating on the test set. --------")
    affinity = adapter(test_features).astype(jt.float32)  # [4305,1496,]
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values  # [4305,403,]
    clip_logits = 100. * test_features @ clip_weights
    mlp_affinity = adapter_mlp(test_features).astype(jt.float32)
    tip_logits = clip_logits + cache_logits * best_alpha + mlp_affinity * best_alpha
    preds = tip_logits.tolist()

    print("任务完成")
    return preds




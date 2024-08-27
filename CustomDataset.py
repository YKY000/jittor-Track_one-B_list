# 读取数据集
from typing import Any, Dict
import os
from PIL import Image
from jittor.dataset import Dataset
import jittor as jt
from jittor import nn
from SEAttention import SEAttention


class TrainCustomDataset(Dataset):
    def __init__(self, features, labels, transform=None) -> None:
        super().__init__()
        self.transform = transform
        self.features = features
        self.labels = labels

    def __getitem__(self, idx):
        img, label = self.features[idx], self.labels[idx]
        if self.transform:
            img, label = self.transform(img), label
        return img, label

    def __len__(self):
        return len(self.features)


class DeepClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(DeepClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, num_classes)  # 最后的全连接层

    def execute(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x


class Classifier(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_classes=4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def execute(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class CustomCLIP(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.dtype = jt.float32
        self.se = SEAttention(channel=512, reduction=8)
        # 确保所有参数都可训练
        for param in self.image_encoder.parameters():
            param.requires_grad = True
        self.logit_scale.requires_grad = True

    def execute(self, image, text_features):
        image = image.astype(self.dtype)  # 使用 Jittor 的类型转换
        image_features = self.image_encoder(image)

        image_features = image_features.unsqueeze(2).unsqueeze(3)
        image_features = self.se(image_features)
        image_features = image_features.squeeze(3).squeeze(2)

        text_features = text_features

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # [16,512,]
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text










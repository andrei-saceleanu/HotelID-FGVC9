import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta + m)
    """

    def __init__(
        self, in_features, out_features, s=30.0, m=0.50, easy_margin=False, device="cuda"
    ):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.device = device

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=self.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output

class HotelIdBackbone(nn.Module):
    """Adapted from michaln's work on Hotel-ID 2021"""

    def __init__(self, embed_size=256, backbone_name="efficientnet_b3"):
        super(HotelIdBackbone, self).__init__()

        self.embed_size = embed_size
        if "vit" in backbone_name:
            self.backbone = timm.create_model(backbone_name, img_size=512, pretrained=False)
        else:
            self.backbone = timm.create_model(backbone_name, pretrained=False)

        in_features = getattr(self.backbone.get_classifier(), "in_features", 768)

        fc_name, _ = list(self.backbone.named_modules())[-1]
        if fc_name == "classifier":
            self.backbone.classifier = nn.Identity()
        elif fc_name == "head.fc":
            self.backbone.head.fc = nn.Identity()
        elif fc_name == "fc":
            self.backbone.fc = nn.Identity()
        elif fc_name == "head.flatten":
            self.backbone.head.fc = nn.Identity()
        elif fc_name == "head":
            self.backbone.head = nn.Identity()

        self.post = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(in_features, self.embed_size * 2), dim=None),
            nn.BatchNorm1d(self.embed_size * 2),
            nn.Dropout(0.2),
            nn.utils.weight_norm(nn.Linear(self.embed_size * 2, self.embed_size)),
            nn.BatchNorm1d(self.embed_size)
        )

    def forward(self, input):
        x = self.backbone(input)
        x = x.view(x.size(0), -1)
        x = self.post(x)
        return x
    
class HotelID(nn.Module):
    def __init__(
        self,
        num_embedding: int,
        num_hotels: int,
        backbone: str,
        **kwargs
    ):
        super(HotelID, self).__init__(**kwargs)

        self.num_embedding = num_embedding
        self.num_hotels = num_hotels
        
        # Embedding layer
        self.embedding_layer = HotelIdBackbone(
            self.num_embedding, backbone
        )

        # Use ArcMargin as our prediction, before the cross-entropy loss
        self.prediction_layer = ArcMarginProduct(
            self.num_embedding,
            self.num_hotels,
            s=30.0,
            m=0.20,
            easy_margin=False
        )

        #self.prediction_layer = nn.Linear(self.num_embedding, self.num_hotels)

    def embed(self, inputs):
        return self.embedding_layer(inputs)

    def forward(self, inputs, labels=None):

        embeddings = self.embed(inputs)
        if labels is not None:
            preds = self.prediction_layer(embeddings, labels)
            return embeddings, preds
        return embeddings

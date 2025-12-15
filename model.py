/oak/stanford/groups/euan/projects/jackos/ecg/git/AI_ECG_HFPEF_8]$ cat model.py 
"""Objects for defining AI-ECG models."""

import torch
from torch import nn
import torchvision
import numpy as np
import sklearn.metrics
import scipy.special
from torchinfo import summary

from model_specs import specs, two_d_specs

class AttentionLayer(nn.Module):
    def __init__(self, attention_type, num_channels):
        super(AttentionLayer, self).__init__()
        if attention_type == "spatial":
            # Spatial attention for 2D inputs
            self.attention = nn.Conv2d(num_channels, 1, kernel_size=7, padding=3)
            self.sigmoid = nn.Sigmoid()
        elif attention_type == "channel":
            # Channel attention for 2D inputs
            self.attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(num_channels, num_channels // 8, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(num_channels // 8, num_channels, kernel_size=1),
                nn.Sigmoid()
            )
        else:
            raise ValueError(f"Unsupported attention type: {attention_type}")

    def forward(self, x):
        if isinstance(self.attention, nn.Conv2d):  # Spatial attention
            attention_weights = self.sigmoid(self.attention(x))
        else:  # Channel attention
            attention_weights = self.attention(x)
        return x * attention_weights

class ECGModel(torch.nn.Module):
    def __init__(self, cfg, num_input_channels, num_outputs, binary, score=None):
        super(ECGModel, self).__init__()
        self.cfg = cfg
        self.out_channels = num_outputs
        self.binary = binary
        self.num_channels_for_adaptive_2d = 12

        if self.cfg["is_2d"]:
            self.num_input_channels = 1
            self.features, in_channels = self.make_2d_conv()
        else:
            self.num_input_channels = num_input_channels
            self.features, in_channels = self.make_conv()

        self.classifier = self.make_fc(in_channels)

        if self.binary:
            self.score = score or (lambda y, yh, loss: sklearn.metrics.roc_auc_score(y, yh))
            w = torch.from_numpy(np.array([cfg["pos_weight"]])).float()
            self.loss = torch.nn.BCEWithLogitsLoss(pos_weight=w)
        else:
            self.score = score or (lambda y, yh, loss: sklearn.metrics.r2_score(y, yh))
            self.loss = torch.nn.MSELoss()

        self.float()

        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(self, flush=True)
        print(f"num params: {num_params}", flush=True)

    def forward(self, x):
        if self.cfg["is_2d"]:
            x = x.unsqueeze(1)
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def train_step(self, x, y):
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        return (y, y_hat, loss)

    def make_conv(self):
        spec = specs[self.cfg["model_type"]][0]
        in_channels = self.num_input_channels

        layers = []
        for d in spec:
            if type(d) == int:
                d = ("C", d, self.cfg["conv_width"], 1)
            elif type(d) == str:
                d = (d)

            if d[0] == "C":
                self.check_length(d, [2, 3, 4])
                if len(d) == 2:
                    d = (d[0], d[1], self.cfg["conv_width"], 1)
                if len(d) == 3:
                    d = (d[0], d[1], d[2], 1)
                conv = nn.Conv1d(in_channels, d[1], kernel_size=d[2], padding=d[3])
                if self.cfg["batch_norm"]:
                    layers += [conv, nn.BatchNorm1d(d[1]), nn.ReLU(inplace=False)]
                else:
                    layers += [conv, nn.ReLU(inplace=False)]
                in_channels = d[1]

            elif d[0] == "attention":
                layers += [AttentionLayer(d[1], in_channels)]

            elif d[0] == "m":
                self.check_length(d, [1, 2])
                if len(d) == 1:
                    d = (d[0], 2)
                layers += [nn.MaxPool1d(kernel_size=d[1], stride=d[1])]

        return nn.Sequential(*layers), in_channels

    def make_2d_conv(self):
        spec = two_d_specs[self.cfg["model_type"]][0]
        in_channels = self.num_input_channels

        layers = []
        for d in spec:
            if type(d) == str:
                d = (d,)
            elif type(d) == int:
                d = ("C", int(d), (int(self.cfg["conv_width"]), int(self.cfg["conv_width"])), (1, 1))
            if d[0] == "l":
                if len(d) == 1:
                    d = (d[0], 64)
                d = ("C", d[1], (self.num_channels_for_adaptive_2d, 1), (0, 0))
            if d[0] == "C":
                if len(d) == 3:
                    d = (d[0], d[1], d[2], (1, 1))
                if type(d[2]) == int:
                    d = (d[0], d[1], (d[2], d[2]), d[3])
                conv = nn.Conv2d(in_channels, d[1], kernel_size=d[2], padding=d[3])
                if self.cfg["batch_norm"]:
                    layers += [conv, nn.BatchNorm2d(d[1]), nn.ReLU(inplace=True)]
                else:
                    layers += [conv, nn.ReLU(inplace=True)]
                in_channels = d[1]
            elif d[0] == "attention":
                if d[1] == "channel":
                    layers += [AttentionLayer("channel", in_channels)]
                elif d[1] == "spatial":
                    layers += [AttentionLayer("spatial", in_channels)]
                else:
                    raise ValueError(f"Unsupported attention type: {d[1]}")
            elif d[0] == "m":
                if len(d) == 1:
                    d = (d[0], (2, 2))
                layers += [nn.MaxPool2d(kernel_size=d[1], stride=d[1])]
            elif d[0] == "a":
                if len(d) == 1:
                    d = (d[0], (3, 3))
                layers += [nn.AdaptiveAvgPool2d(output_size=d[1])]
                in_channels = np.prod(d[1]) * in_channels
        return nn.Sequential(*layers), in_channels

    def make_fc(self, in_channels):
        if self.cfg["is_2d"]:
            spec = two_d_specs[self.cfg["model_type"]][1]
        else:
            spec = specs[self.cfg["model_type"]][1]

        spec = spec.copy()
        spec.append(self.out_channels)

        layers = []
        for d in spec:
            if type(d) == int:
                layers.append(nn.Linear(in_channels, d))
                in_channels = d
            elif d == "r":
                layers.append(nn.ReLU(inplace=True))
            elif d == "d":
                layers.append(nn.Dropout(self.cfg["drop_prob"]))
            elif d == "b":
                layers.append(nn.BatchNorm1d(in_channels))
        return nn.Sequential(*layers)

    def check_length(self, obj, lengths):
        assert len(obj) in lengths, f"{obj} is malformed"

    def make_layer(self, in_channels, out_channels, n_blocks, kernel_size=3, block=None, batch_norm=False):
        if block is None:
            block = Block
        blocks = [block(in_channels, out_channels, average_pool=True, kernel_size=kernel_size, batch_norm=batch_norm)]
        blocks += [block(out_channels, out_channels, kernel_size=kernel_size, batch_norm=batch_norm) for _ in range(1, n_blocks)]
        return nn.Sequential(*blocks)

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, average_pool=False, batch_norm=True):
        super(Block, self).__init__()
        if batch_norm:
            self.backbone = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, padding=1, kernel_size=kernel_size, stride=(2 if average_pool else 1)),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=False),
                nn.Conv1d(out_channels, out_channels, padding=1, kernel_size=kernel_size),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=False)
            )
        else:
            self.backbone = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, padding=1, kernel_size=kernel_size, stride=(2 if average_pool else 1)),
                nn.ReLU(inplace=False),
                nn.Conv1d(out_channels, out_channels, padding=1, kernel_size=kernel_size),
                nn.ReLU(inplace=False)
            )

        self.downsample = None

        if average_pool:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=2)
            )

    def forward(self, x):
        identity = x
        if self.downsample:
            identity = self.downsample(identity)
        x = identity + self.backbone(x)
        return x

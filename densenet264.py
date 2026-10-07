import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import trange
from collections import OrderedDict

# -------------------
# DenseNet-264
# -------------------
class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size=4, drop_rate=0.0):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate,
                               kernel_size=1, stride=1, bias=False)
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.conv1(self.relu1(self.norm1(x)))
        out = self.conv2(self.relu2(self.norm2(out)))
        if self.drop_rate > 0.0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return torch.cat([x, out], 1)

class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, growth_rate, bn_size=4, drop_rate=0.0):
        super().__init__()
        layers = []
        num_features = num_input_features
        for i in range(num_layers):
            layer = _DenseLayer(num_features, growth_rate, bn_size, drop_rate)
            layers.append(layer)
            num_features += growth_rate
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class _Transition(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features,
                              kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(self.relu(self.norm(x)))
        x = self.pool(x)
        return x

class DenseNet264(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, growth_rate=32,
                 block_config=(6, 12, 64, 48), bn_size=4, compression=0.5, drop_rate=0.0):
        super().__init__()

        # Initial convolution
        num_init_features = 64
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Dense blocks and transitions
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers, num_features, growth_rate, bn_size, drop_rate)
            self.features.add_module(f'denseblock{i+1}', block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                out_features = int(num_features * compression)
                trans = _Transition(num_features, out_features)
                self.features.add_module(f'transition{i+1}', trans)
                num_features = out_features

        # Final batch norm
        self.features.add_module('norm_final', nn.BatchNorm2d(num_features))
        self.relu_final = nn.ReLU(inplace=True)

        # Classifier
        self.classifier = nn.Linear(num_features, num_classes)

        # He init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.relu_final(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# -------------------
# Training function
# -------------------
def train_one_round(train_dir, test_dir, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # standard input size
        transforms.ToTensor()
    ])

    train_data = datasets.ImageFolder(root=train_dir, transform=transform)
    test_data = datasets.ImageFolder(root=test_dir, transform=transform)

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

    model = DenseNet264(in_channels=3, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train
    start_train = time.time()
    model.train()
    for _ in range(10):  # 10 epochs
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    end_train = time.time()
    train_time = end_train - start_train

    # Test
    start_test = time.time()
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    end_test = time.time()
    test_time = end_test - start_test

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return acc, prec, rec, f1, train_time, test_time

# -------------------
# Main Loop
# -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

round_dirs = [
    ("round_1_test_rp", "round_1_train_rp"),
    ("round_2_test_rp", "round_2_train_rp"),
    ("round_3_test_rp", "round_3_train_rp"),
    ("round_4_test_rp", "round_4_train_rp"),
    ("round_5_test_rp", "round_5_train_rp"),
    ("round_6_test_rp", "round_6_train_rp"),
    ("round_7_test_rp", "round_7_train_rp"),
    ("round_8_test_rp", "round_8_train_rp"),
    ("round_9_test_rp", "round_9_train_rp"),
    ("round_10_test_rp", "round_10_train_rp"),
    """
    ("round_11_test_rp", "round_11_train_rp"),
    ("round_12_test_rp", "round_12_train_rp"),
    ("round_13_test_rp", "round_13_train_rp"),
    ("round_14_test_rp", "round_14_train_rp"),
    ("round_15_test_rp", "round_15_train_rp"),
    ("round_16_test_rp", "round_16_train_rp"),
    ("round_17_test_rp", "round_17_train_rp"),
    ("round_18_test_rp", "round_18_train_rp"),
    ("round_19_test_rp", "round_19_train_rp"),
    ("round_20_test_rp", "round_20_train_rp"),
    ("round_21_test_rp", "round_21_train_rp"),
    ("round_22_test_rp", "round_22_train_rp"),
    """
]

for i in trange(len(round_dirs), desc="Rounds", ncols=100):
    train_dir, test_dir = round_dirs[i]
    acc, prec, rec, f1, t_time, tst_time = train_one_round(train_dir, test_dir, device)
    print(f"Round {i+1} — "
          f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, "
          f"Train Time: {t_time:.2f}s, Test Time: {tst_time:.2f}s")

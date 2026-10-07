import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import trange

# -------------------
# ResNet-152
# -------------------
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Optional: initialize weights (common practice)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# -------------------
# Training function
# -------------------
def train_one_round(train_dir, test_dir, device):
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor()
    ])

    train_data = datasets.ImageFolder(root=train_dir, transform=transform)
    test_data = datasets.ImageFolder(root=test_dir, transform=transform)

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

    model = ResNet(Bottleneck, [3, 8, 36, 3], num_classes=2).to(device)  # ResNet-152
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
    test_dir, train_dir = round_dirs[i]
    acc, prec, rec, f1, t_time, tst_time = train_one_round(train_dir, test_dir, device)
    print(f"Round {i+1} — "
          f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, "
          f"Train Time: {t_time:.2f}s, Test Time: {tst_time:.2f}s")

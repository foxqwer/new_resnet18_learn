import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image
import os

EPOCHS = 5

class FruitVegetableDataset():
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.images = self._load_images()

    # _load_images 函数用于加载图片并返回一个列表，其中每个元素是一个元组 (图片路径, 类别索引)
    # 索引类别是把其中的类，从0到35依次标号，这样方便后续的训练和测试。比如apple标记为0，banana标记为1，
    # 以此类推，直到35。
    def _load_images(self):
        images = []
        for cls in self.classes:
            cls_dir = os.path.join(self.root_dir, cls)
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                images.append((img_path, self.class_to_idx[cls]))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# 数据预处理
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图片大小
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize(mean=[0.47029597, 0.43234787, 0.32367933], std=[0.37582668, 0.35999426, 0.34868242])  # 归一化
])

transform_text = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图片大小
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize(mean=[0.46606797, 0.42894393, 0.32173076], std=[0.37519165, 0.3603049, 0.34899382])  # 归一化
])

# 创建训练集和测试集的 Dataset 对象
train_dataset = FruitVegetableDataset(root_dir='.\\train', transform=transform_train)
test_dataset = FruitVegetableDataset(root_dir='.\\test', transform=transform_text)

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 定义模型
# 加载预训练的 ResNet-18 模型
model = models.resnet18(pretrained=True)

# 修改最后的全连接层，使其输出类别数与数据集类别数一致
num_classes = len(train_dataset.class_to_idx)  # 数据集的类别数
model.fc = nn.Linear(model.fc.in_features, num_classes)

# 将模型移动到 GPU（如果可用）
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)




# 初始化模型

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练函数
def train(model, train_loader, criterion, optimizer, epochs=EPOCHS):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:  # 每100个batch打印一次损失
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')
                running_loss = 0.0

# 开始训练
train(model, train_loader, criterion, optimizer, epochs=EPOCHS)

# 测试函数
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the 10000 test images: {100 * correct / total:.2f}%')

# 开始测试
test(model, test_loader)

# 保存模型
torch.save(model.state_dict(), 'resnet18.pth')



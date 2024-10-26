import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import matplotlib.pyplot as plt 
from torchsummary import summary
import json
from model import CNNKAN

model = CNNKAN(num_classes=2).to(device)

# Need to explore various Optimizer and optimize the Learning Rate.

optimizer = optim.AdamW(model.parameters(), lr=0.00005, weight_decay=1e-3)

batch_size = 32

nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers

data_transform = {
    "train": transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.RandomResizedCrop(size=(64, 64), scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.1)]),
        
    "val": transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

train_dataset = datasets.ImageFolder(root="/kaggle/working/dataset/train",
                                         transform=data_transform["train"])

train_num = len(train_dataset)

# Calculate class weights
class_counts = [0] * 2  # Assuming 2 classes
for _, label in train_dataset:
    class_counts[label] += 1
    
class_weights = [3986.0/count for count in class_counts]

print("Class Weights")
print(class_weights)

weights = [class_weights[label] for _, label in train_dataset]
sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))

flower_list = train_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in flower_list.items())
# write dict into json file
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=batch_size,
                                            sampler=sampler,
                                            num_workers=nw)


test_dataset = datasets.ImageFolder(root="/kaggle/working/dataset/val",
                                        transform=data_transform["val"])

test_num = len(test_dataset)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False,
                                          num_workers=nw)

print("using {} images for training, {} images for validation.".format(train_num,
                                                                        test_num))



def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def evaluate(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.CrossEntropyLoss()(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    
    accuracy = 100 * correct / len(test_loader.dataset)
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy}%)\n')
    
    return accuracy

best_accuracy = 0

for epoch in range(401):
    
    train(model, device, train_loader, optimizer, epoch)
    current_accuracy = evaluate(model, device, test_loader)
    
    if current_accuracy > best_accuracy:
        best_accuracy = current_accuracy
        print(f"Validation Accuracy Improved to {best_accuracy}")
        torch.save(model.state_dict(), 'ConvKAN_best_weights.pth')
        
    print(f"Current Best Validation Accuracy: {best_accuracy}\n")
        
        
    

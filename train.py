from model import ImageTextModel
from dataset import ImageTextDataset
import pandas as pd
from torchvision import transforms
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ImageTextModel(num_classes=20).to(device)
total_num = sum(p.numel() for p in model.parameters())
print("Total number of parameters:", total_num)
data_given = pd.read_csv("data/train.csv", on_bad_lines='skip')

train_df, test_df = train_test_split(data_given, test_size = 0.2)

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Train on the whole training set
train_data = ImageTextDataset(data_given, "/content/COMP5329S1A2Dataset/data", transform=transform)
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
model = ImageTextModel(num_classes=20).to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

epochs = 4
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    ImageTextModel.train_loop(train_dataloader, model, loss_fn, optimizer)
print("Model training completed")
import os
from PIL import Image
from torch.utils.data import Dataset
from transformers import BertTokenizer
import torch

class ImageTextDataset(Dataset):
    def __init__(self, data_frame, image_dir, transform=None, max_length=128):
        self.data_frame = data_frame
        self.image_dir = image_dir
        self.transform = transform
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, index):
        row = self.data_frame.iloc[index]

        # Image processing
        image_path = os.path.join(self.image_dir, row.ImageID)
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Text processing
        text = row.Caption
        tokenized = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']

        # Label processing
        label_indices = list(map(int, row.Labels.strip('[]').split()))
        labels = torch.zeros(20)
        labels[label_indices] = 1

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'image': image,
            'labels': labels,
            'image_path': image_path
        }
import torchvision
from transformers import BertModel
import torch
from torch import nn
class ImageTextModel(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', num_classes=2):
        super(ImageTextModel, self).__init__()
        self.text_model = BertModel.from_pretrained(bert_model_name)  # Load pre-trained BERT model
        self.image_model = torchvision.models.resnet50(pretrained=True)  # Load pre-trained ResNet model
        self.image_model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.image_model.fc = nn.Linear(2048, 768)
        self.classifier = nn.Linear(768, num_classes)  # Classification layer

    def forward(self, input_ids, attention_mask, image):
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)  # Process text
        text_features = text_outputs.last_hidden_state[:, 0]
        image_features = self.image_model(image)  # Process image
        combined_features = text_features + image_features
        output = self.classifier(combined_features)  # Final classification
        return output
    
    def train_loop(dataloader, model, loss_fn, optimizer):
        dataset_size = len(dataloader.dataset)
        model.train()  # Set model to training mode

        for batch_idx, batch_data in enumerate(dataloader):
            predictions = model(batch_data['ids'].to(device),
                                batch_data['mask'].to(device),
                                batch_data['image'].to(device))
            loss = loss_fn(predictions, batch_data['labels'].to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:  # Print loss every 100 batches
                current = batch_idx * len(batch_data['image'])
                print(f"loss: {loss.item():>7f}  [{current:>5d}/{dataset_size:>5d}]")
        
    def test_loop(dataloader, model, loss_fn, test=True):
        dataset_size = len(dataloader.dataset)
        test_loss, correct = 0, 0
        all_predictions = []
        all_labels = []
        all_image_paths = []

        model.eval()  # Set model to evaluation mode

        with torch.no_grad():
            for batch_data in dataloader:
                predictions = model(batch_data['ids'].to(device),
                                    batch_data['mask'].to(device),
                                    batch_data['image'].to(device))  # Make predictions
                all_image_paths.append(batch_data['image_names'])
                for i in range(len(predictions)):
                    if test:
                        test_loss += loss_fn(predictions[i], batch_data['labels'][i].to(device)).item()
                        if ((predictions[i] > 0) == batch_data['labels'][i].to(device)).all():
                            correct += 1
                        all_labels.append(batch_data['labels'][i])
                    pred_binary = torch.where(predictions[i] < 0, torch.tensor(0), torch.tensor(1))
                    all_predictions.append(pred_binary.cpu())  # Store predictions

        test_loss /= dataset_size
        correct /= dataset_size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")  # Print results
        return all_predictions, all_labels, all_image_paths
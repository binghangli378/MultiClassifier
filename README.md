# Image-Text Multimodal Classification

This project is designed for multimodal classification based on both image and text inputs. It uses a custom PyTorch model to process the data and perform training, validation, and testing. The dataset consists of images and associated textual labels.

## Dataset

The dataset is not included in this repository. You can download the dataset from [this Google Drive link](https://drive.google.com/file/d/1Y9JpEf_Y22bZYqm6YbmFkorjmG2m_9bo/view?usp=sharing).

### Data Format
- **train.csv**: The training data, which contains image paths and corresponding text labels.
- **test.csv**: The testing data, containing image paths and dummy text labels (which will be predicted by the model).
- **data/**: A directory containing the images used for training and testing.

Once downloaded, place the dataset in the `data/` directory

## Dependencies

To install the necessary dependencies, you can use the provided `requirements.txt` file.

```bash
pip install -r requirements.txt

## Training

To train the model, run the train.py script. This will load the dataset, initialize the model, and train it for a set number of epochs.

```bash
python train.py



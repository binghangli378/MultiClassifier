import os
import pandas as pd
import numpy as np
from model import ImageTextModel
from dataset import ImageTextDataset
from torch.utils.data import DataLoader
test_file = pd.read_csv('data/COMP5329S1A2Dataset/test.csv', on_bad_lines='skip')
test_file['Labels'] = '0'
# Fill bad lines
pd_arr1 = test_file[:6889]
pd_arr2 = test_file[6889:]
pd_arr1.loc[6889] = ['36889.jpg', 'Stop sign with added war" annotation at an intersection."', '0']
test_file = pd_arr1.append(pd_arr2, ignore_index=True)
test_data = ImageTextDataset(test_file, '/content/COMP5329S1A2Dataset/data', transform=transform)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
pred, rightlabel, all_numbers = ImageTextModel.test_loop(test_dataloader, model, loss_fn, test=False)
y_pred = [t.numpy() for t in pred]

image_path = []
for batch in all_numbers:
    image_path.append([os.path.basename(path) for path in batch])
image_names = [item for sublist in image_path for item in sublist]
y_pred_indices = [np.where(arr==1)[0] for arr in y_pred]
y_pred_strings = [' '.join(map(str, arr)) for arr in y_pred_indices]
predict_output = pd.DataFrame({'ImageID': image_names, 'Labels': y_pred_strings})
predict_output['ImageID'] = predict_output['ImageID'].str.extract('(\d+)').astype(int)
predict_output = predict_output.sort_values('ImageID')
predict_output['ImageID'] = predict_output['ImageID'].astype(str) + '.jpg'
print(predict_output)
predict_output.to_csv('predict_output.csv', index=False)

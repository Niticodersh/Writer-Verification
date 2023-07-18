#Importing libraries -----------------------------------------------------------------------------------------------------------------
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import DataLoader
import random
from PIL import Image
import PIL.ImageOps
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import dataloader
import torchvision.utils
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Dataset
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import euclidean
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib
from sklearn.svm import SVC
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

#path ------------------------------------------------------------------------------------------------------------------------------------
siamese_model_path="/content/drive/MyDrive/dataset/SaraSwati_Writes_Final_Model.pth"
knn_model_path='/content/drive/MyDrive/dataset/knn_model.pkl'
test_data_path="/content/drive/MyDrive/dataset/test.csv"
string_to_concat = "/content/drive/MyDrive/dataset/semi_test/"
path_to_save_submission_file='/content/drive/MyDrive/dataset/SSWrites_Semi_final_Submission.csv'

# Siamses Architectue ---------------------------------------------------------------------------------------------------------------------
class SiameseNetwork(nn.Module):
  def __init__(self):
    super(SiameseNetwork, self).__init__()

    self.conv1 = nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=1)
    self.bn1 = nn.BatchNorm2d(num_features=96)
    self.maxpool1 = nn.MaxPool2d(kernel_size=3,stride= 2)
    self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
    self.bn2 = nn.BatchNorm2d(num_features=256)
    self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
    self.dropout1 = nn.Dropout(p=0.3)
    self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
    self.conv4 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
    self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)
    self.dropout2 = nn.Dropout(p=0.3)
    self.fc1= nn.Linear(in_features=108800, out_features=1024)
    self.dropout3 = nn.Dropout(p=0.5)
    self.fc2= nn.Linear(in_features=1024,out_features=128)
    self.relu = nn.ReLU(inplace = True)

  def forward(self, x1, x2):
    x1 = self.conv1(x1)
    x1 = self.relu(x1)
    x1 = self.bn1(x1)
    x1 = self.maxpool1(x1)
    x1 = self.conv2(x1)
    x1 = self.relu(x1)
    x1 = self.bn2(x1)
    x1 = self.maxpool2(x1)
    x1 = self.dropout1(x1)
    x1 = self.conv3(x1)
    x1 = self.relu(x1)
    x1 = self.conv4(x1)
    x1 = self.relu(x1)
    x1 = self.maxpool3(x1)
    x1 = self.dropout2(x1)
    x1 = x1.view(x1.size()[0],-1)
    x1 = self.fc1(x1)
    x1 = self.relu(x1)
    x1 = self.dropout3(x1)
    x1 = self.fc2(x1)

    x2 = self.conv1(x2)
    x2 = self.relu(x2)
    x2 = self.bn1(x2)
    x2 = self.maxpool1(x2)
    x2 = self.conv2(x2)
    x2 = self.relu(x2)
    x2 = self.bn2(x2)
    x2 = self.maxpool2(x2)
    x2 = self.dropout1(x2)
    x2 = self.conv3(x2)
    x2 = self.relu(x2)
    x2 = self.conv4(x2)
    x2 = self.relu(x2)
    x2 = self.maxpool3(x2)
    x2 = self.dropout2(x2)
    x2 = x2.view(x2.size()[0],-1)
    x2 = self.fc1(x2)
    x2 = self.relu(x2)
    x2 = self.dropout3(x2)
    x2 = self.fc2(x2)
    return x1, x2

# Transformation to images -------------------------------------------------------------------------------------------------------------------------
transform=transforms.Compose(
        [transforms.Resize((155, 220)), transforms.ToTensor()])

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Extract output features of siamese model -----------------------------------------------------------------------------------------------------------

def extract_features(model,data):  #type(data)-->Dataframe
    model.eval()
    features=[]
    # Iterate over the rows of the DataFrame starting from index 1600
    for sample in data.iloc[:, :].values:
        # Load the image using PIL
        image1 = Image.open(sample[0])
        image2 = Image.open(sample[1])
        # print(sample[0],sample[1])
        # Apply the transformation pipeline
        transformed_image1 = transform(image1)
        transformed_image2 = transform(image2)

        # Move the tensors to the appropriate device
        transformed_image1 = transformed_image1.to(device)
        transformed_image2 = transformed_image2.to(device)

        # Add an extra dimension to match the model input shape
        transformed_image1 = transformed_image1.unsqueeze(0)
        transformed_image2 = transformed_image2.unsqueeze(0)

        # Forward pass through the model
        with torch.no_grad():
            feature=[]
            output1, output2= model(transformed_image1, transformed_image2)
            output1 = output1/torch.norm(output1)
            output2 = output2/torch.norm(output2)
            output1 = output1.cpu().numpy()
            output2 = output2.cpu().numpy()
            feature.append(output1)
            feature.append(output2)
        features.append(feature)
    return features

#Prediction using KNN ---------------------------------------------------------------------------------------------------------------------------------
def knn_predict(test_features):
    knn = joblib.load(knn_model_path)
    predicted_labels = knn.predict(test_features)
    probabilities = knn.predict_proba(test_features)[:,1]
    return predicted_labels, probabilities

#Loading the trained siamese model ---------------------------------------------------------------------------------------------------------------------
net = SiameseNetwork().to(device)
# Load the saved state dictionary
state_dict = torch.load(siamese_model_path)
# Load the state dictionary into the model
net.load_state_dict(state_dict)

# Load the test data ------------------------------------------------------------------------------------------------------------------------------------
test_data=pd.read_csv(test_data_path)
test_data_copy = test_data.copy()
# Concatenate string to the start of each testue in columns 1 and 2
test_data_copy.iloc[:, 0:2] = string_to_concat + test_data_copy.iloc[:, 0:2].astype(str)

test_features = extract_features(net,test_data_copy)

test_features=np.array(test_features)
test_features_reshaped = np.squeeze(test_features)
test_features_reshaped = np.reshape(test_features_reshaped, test_features.shape)
test_features_flattened = np.reshape(test_features_reshaped, (test_features.shape[0], -1))

predicted_test_labels, test_probabilities = knn_predict(np.array(test_features_flattened))

print("Predicted_label",predicted_test_labels)
print("Predicted_Probabilities",test_probabilities)

#Save the predictions -----------------------------------------------------------------------------------------------------------------------------------
submission = pd.DataFrame()
submission['id'] = test_data['img1_name']+'_'+test_data['img2_name']
submission['proba'] = test_probabilities
submission.to_csv(path_to_save_submission_file, index=False)

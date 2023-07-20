#Importing Libraries --------------------------------------------------------------------------------------------------------------------
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
import pickle
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
import joblib
from scipy.spatial.distance import euclidean
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

#Path --------------------------------------------------------------------------------------------------------------------------------
Training_dataset_path="/content/drive/MyDrive/dataset/Training_Dataset.csv"
siamese_model_dict_path="/content/drive/MyDrive/dataset/SaraSwati_Writes_Final_Model.pth"
siamese_model_path="/content/drive/MyDrive/dataset/SaraSwati_Writes_Final_Model.pkl"
validation_data_path="/content/drive/MyDrive/dataset/val.csv"
string_to_concat = "/content/drive/MyDrive/dataset/val/"
knn_model_path= '/content/drive/MyDrive/dataset/knn_model.pkl'

#Load Training Dataset ---------------------------------------------------------------------------------------------------------------
data=pd.read_csv(Training_dataset_path)
train_data = data.iloc[:16000,:]
eval_data = data.iloc[16000:,:] #This is a part of training_data only which we have used for early stopping
#Building Siamese Network Architecture -----------------------------------------------------------------------------------------------
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


# Contrastive Loss Function ----------------------------------------------------------------------------------------------------------

class ContrastiveLoss(nn.Module):
    "Contrastive loss function"

    def __init__(self, margin=1.0, constant1=1, constant2=2):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.constant1 = torch.tensor(constant1, dtype=torch.float32)
        self.constant2 = torch.tensor(constant2, dtype=torch.float32)

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, p=2)
        loss_contrastive = torch.mean(
            (label) * self.constant1 * torch.pow(euclidean_distance, 2)
            + (1 - label) * self.constant2
            * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )

        return loss_contrastive
    
# Creating Dataloader for training and validation -------------------------------------------------------------------------------------

from torchvision.transforms import ToTensor

class SiameseDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Load the image using OpenCV
        image1 = Image.open(self.df.iloc[idx, 0])
        image2 = Image.open(self.df.iloc[idx, 1])

        # Apply the transformation pipeline if provided
        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        label = self.df.iloc[idx, 2]

        return image1, image2, torch.tensor(label, dtype=torch.float32)

transform=transforms.Compose(
        [transforms.Resize((155, 220)), transforms.ToTensor()])

 # Load the dataset as pytorch tensors using dataloader
train_dataloader = DataLoader(SiameseDataset(train_data, transform=transform),
                        shuffle=True,
                        num_workers=2,
                        batch_size=32)

 # Load the dataset as pytorch tensors using dataloader
eval_dataloader = DataLoader(SiameseDataset(eval_data, transform=transform),
                        shuffle=True,
                        num_workers=2,
                        batch_size=32)

#Code for training ------------------------------------------------------------------------------------------------------------------------

# Declare Siamese Network
net = SiameseNetwork().cuda()
# Declare Loss Function
criterion = ContrastiveLoss()
# Declare Optimizer
optimizer = torch.optim.Adam(net.parameters(), lr = 1e-4, weight_decay=0.0005)
scheduler = ExponentialLR(optimizer, gamma=0.9)

#train the model
def train(train_dataloader):
    loss=[]
    counter=[]
    iteration_number = 0
    for i, data in enumerate(train_dataloader,0):
      img0, img1 , label = data
      img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
      optimizer.zero_grad()
      output1,output2 = net(img0,img1)
      loss_contrastive = criterion(output1,output2,label)
      loss_contrastive.backward()
      optimizer.step()
      loss.append(loss_contrastive.item())
    loss = np.array(loss)
    return loss.mean()/len(train_dataloader)


def eval(eval_dataloader):
    loss=[]
    counter=[]
    iteration_number = 0
    for i, data in enumerate(eval_dataloader,0):
      img0, img1 , label = data
      img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
      output1,output2 = net(img0,img1)
      loss_contrastive = criterion(output1,output2,label)
      loss.append(loss_contrastive.item())
    loss = np.array(loss)
    return loss.mean()/len(eval_dataloader)

best_eval_loss = 9999
for epoch in range(1,30):
  train_loss = train(train_dataloader)
  eval_loss = eval(eval_dataloader)
  print("Epoch:",epoch)
  print(f"Training loss{train_loss}")
  print("-"*10)
  print(f"Eval loss{eval_loss}")
  scheduler.step()

  if eval_loss<best_eval_loss:
    best_eval_loss = eval_loss
    print("-"*10)
    print(f"Best Eval loss{best_eval_loss}")
    torch.save(net.state_dict(), siamese_model_dict_path)
    print("Model Saved Successfully")

with open(siamese_model_path, 'wb') as f:
    pickle.dump(net, f)

print("Model saved as .pkl successfully.")
# Extracting output features of siamese network --------------------------------------------------------------------------------------------------

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
train_features = extract_features(net,train_data)
train_labels = train_data['label']

val_data=pd.read_csv(validation_data_path)
val_data_copy = val_data.copy()

# Concatenate string to the start of each value in columns 1 and 2
val_data_copy.iloc[:, 0:2] = string_to_concat + val_data_copy.iloc[:, 0:2].astype(str)

val_features = extract_features(net,val_data_copy)

train_features=np.array(train_features)
val_features=np.array(val_features)

val_labels = np.asarray(val_data['label'])

train_features_reshaped = np.squeeze(train_features)
train_features_reshaped = np.reshape(train_features_reshaped, train_features.shape)
train_features_flattened = np.reshape(train_features_reshaped, (train_features.shape[0], -1))
# train_features_flattened.shape

val_features_reshaped = np.squeeze(val_features)
val_features_reshaped = np.reshape(val_features_reshaped, val_features.shape)
val_features_flattened = np.reshape(val_features_reshaped, (val_features.shape[0], -1))
# val_features_flattened.shape

#Classification using KNN ------------------------------------------------------------------------------------------------------------------------------------
def knn_fit(train_features, train_labels,k):
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn.fit(train_features, train_labels)
    joblib.dump(knn,knn_model_path)
    
def knn_predict(test_features):
    knn = joblib.load(knn_model_path)
    predicted_labels = knn.predict(test_features)
    probabilities = knn.predict_proba(test_features)[:,1]
    return predicted_labels, probabilities

knn_fit(np.array(train_features_flattened), np.array(train_labels),100)
predicted_labels1, probabilities1 = knn_predict( np.array(val_features_flattened))
f1_knn = f1_score(val_labels, predicted_labels1)
auc_knn = roc_auc_score(val_labels, probabilities1)
acc_knn = accuracy_score(val_labels, predicted_labels1)

print("F1 Score:", f1_knn)
print("AUC:", auc_knn)
print("Accuracy",acc_knn)

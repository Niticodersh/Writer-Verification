# NCVPRIPG Writer Verification  
# TEAM: SSWrites
Kaggle: [https://www.kaggle.com/nitishbhardwajiitj](https://www.kaggle.com/nitishbhardwajiitj)
---------------------------------------------------------------------------------------------------------------------------------------------------------------------- 

This GitHub repository aims to identify whether a given pair of handwritten text samples was written by the same person or two different individuals. It provides a solution for verifying the authenticity of handwritten text samples, making it invaluable for real-world applications such as forensic analysis, document authentication, and signature verification systems.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Table of Contents

1. [Dataset](#dataset)
2. [Training Model](#training-model)
3. [Inference Model](#inference-model)
4. [Model Checkpoints](#model-checkpoints)
5. [Codebase](#codebase)
6. [Requirement.txt](#requirement)
7. [Reference](#reference)

----------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Dataset <a name="dataset"></a>

The dataset used in this project was made available for the NCVPRIG Competition. However, due to the competition's terms and conditions, the actual dataset cannot be shared. We encourage you to use any other suitable dataset for your project.

If you are working with a dataset that is not well-maintained, such as having folders where each folder contains images of the same writers, you can utilize the `training_dataset.py` code provided in this repository. This code will help you create a properly organized CSV file from your dataset. We also recommend reviewing the `training_dataset.py` code to gain insights on how to manage and organize your dataset effectively, even in other scenarios where the dataset may not be well-organized.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Training Model <a name="training-model"></a>

If you want to train the model from scratch, follow these steps:

1. Add the path of your training and validation datasets in the `training_model.py` file.
2. Run the `training_model.py` script to initiate the training process.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Inference Model <a name="inference-model"></a>

If you want to use the pre-trained model directly, follow these steps:

1. Add the path of your test data in the `Inference_model.py` file.
2. Run the `Inference_model.py` script to perform inference using the pre-trained model.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Model Checkpoints <a name="model-checkpoints"></a>

Pre-trained models can be found in the following Google Drive folder:

[Trained models](https://drive.google.com/drive/folders/1GY2brp7-rYLxwLa6WBMvyC_SY3cjr1Cv?usp=sharing)

If you want to use the pre-trained model, download the models from the shared Google Drive folder and save them to your local location. Additionally, update the path location of the model in the code accordingly.

In this project, we employed a Siamese Network for feature extraction, and these features were subsequently utilized by a KNN model for classification. The `training_model.py` and `Inference_model.py` solely employ the KNN approach. However, we also explored alternative classification models such as SVC, AdaBoost, and ensemble methods. If you wish to examine predictions using these models, simply copy and paste their respective code snippets from the `Codebase.ipynb` file.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Codebase <a name="codebase"></a>

The codebase for this project consists of the following files:

- `training_model.py`: Script for training the model from scratch.
- `Inference_model.py`: Script for performing inference using the pre-trained model.
- `Codebase.ipynb`: Jupyter Notebook containing the complete code of this project. We encourage you to go through it to understand the work done.
- `training_dataset.py`: Script for creating an organized dataset.
- `requirements.txt`: Required packages and dependencies.

Feel free to explore and modify the codebase to suit your specific requirements.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Requirement.txt <a name="requirement"></a>
To install the required libraries, run the following command in your terminal:

```bash
pip install -r requirements.txt
```
----------------------------------------------------------------------------------------------------------------------------------------------------------------------

## References <a name="reference"></a>

- Deya, S., Dutta, A., Toledoa, J. I., Ghosha, S. K., Llados, J., & Pal, U. (Year). SigNet: Convolutional Siamese Network for Writer Independent Offline Signature Verification. _Pattern Recognition Letters_. Retrieved from Journal Homepage:[www.elsevier.com](https://www.elsevier.com/en-in)


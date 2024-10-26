# Kolmogorov-Arnold Networks with CNNs for Image Classification of Stars and Galaxies
Welcome to our repository, combining the power of Kolmogorov-Arnold Networks and Convolutional Neural Networks (CNNs) to classify stars and galaxies using an astronomical dataset. This project pushes the boundaries of computer vision and astronomical research, offering a unique approach to stellar classification.

# 📌 Overview
This repository implements lightweight Kolmogorov-Arnold Network fused with CNNs to classify stellar sources such as stars and galaxies. The dataset used for this project was generated at the Aryabhatta Research Institute of Observational Sciences (ARIES), Nainital, India, utilizing their 1.3m telescope. Special thanks to github user https://github.com/jakariaemon/CNN-KAN/ who provided some of the code used in this proyect. For a full detailed presentation of the model's training and evaluation, please visit its official Kaggle notebook at https://www.kaggle.com/code/diegoexe/kan-for-star-galaxy-classification-90-45-val-acc/

# 🗂 Dataset Description

The dataset was created as part of a project at ARIES and involves images captured by the 1.3m telescope situated in Devasthal, Nainital, India.

Classes: Stars and Galaxies

Dataset Size: 3986 total images, 3044 stars and 942 galaxies

Image Size: 64 x 64 pixels

Author: Divyansh Agrawal

Stars

![grb0422a_01_94_131_6](https://github.com/user-attachments/assets/9d019f16-2f3e-4468-ae3f-48d2e1596cd2)
![grb0422a_01_131_1127_6](https://github.com/user-attachments/assets/8e094ee6-7db0-4521-bfe0-cfd772f7f9a7)
![grb0422a_01_129_476_6](https://github.com/user-attachments/assets/a3d62ef8-d6fa-4844-ba2e-45a365845d7a)
![grb0422a_01_120_1260_6](https://github.com/user-attachments/assets/57a1acd8-7153-45d7-ba8f-e8a9b642fb00)
![grb0422a_01_119_679_6](https://github.com/user-attachments/assets/c4208bb7-9c15-466b-be21-eeded59fd913)
![grb0422a_01_117_1527_6](https://github.com/user-attachments/assets/9d500ec6-3bf5-4499-bea0-9e743033c313)


Galaxies

![J1530p2310_r_01_92_1222_3](https://github.com/user-attachments/assets/d4f40d1c-bbbc-4799-b405-d2fb10f0a271)
![J1530p2310_r_01_356_281_3](https://github.com/user-attachments/assets/0d23042b-1825-4890-884b-c05752e1f902)
![J1530p2310_r_01_344_962_3](https://github.com/user-attachments/assets/1c652745-07d3-4a42-8228-7853875662bd)
![J1530p2310_r_01_342_202_3](https://github.com/user-attachments/assets/27f55e01-476a-4053-87b1-93f0255fd154)
![J1530p2310_r_01_336_941_3](https://github.com/user-attachments/assets/840782a4-75b7-4d9e-9355-4d394f630684)
![J1530p2310_r_01_301_607_3](https://github.com/user-attachments/assets/ac3643cd-77e8-46d6-9db1-124acfd8fedd)


Find more information about the dataset https://www.kaggle.com/datasets/divyansh22/dummy-astronomy-data

# Results

As described in the Kaggle notebook in the link attached, the dataset was set up for a random split of the data with a ratio of 80:20 (80% for training and 20% for validation). The main challenge of the classification task lies on the low resolution of the images presented and the considerable imbalance presented by the classes in the dataset. Therefore heavy data augmentation and weights corresponding to the classes were needed to enhance the model's training, this concluding on a 90.45% validation accuracy

![image](https://github.com/user-attachments/assets/57f5293c-b7e5-4f3a-a149-1a900bea7952)





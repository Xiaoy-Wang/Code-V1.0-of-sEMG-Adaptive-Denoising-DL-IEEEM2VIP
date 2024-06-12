Codes for the conference paper titled "Exploration of Adaptive sEMG Denoising and Deep Learning Paradigm and their Application in Lower Limb Motion Recognition" (IEEEM2VIP, 2024).

Install:

Use "Install.ipynb" to create the environment and dependencies, or install the required dependencies using "pip install requirements.txt".

Usage:
1.	See "processing.ipynb" file in "preProcessing" folder to preprocess and denoise the sEMG signals from SIAT-LLMD dataset and  convert them into training and testing sample sets.
2.	See "modelTest.ipynb" filel in "models" folder for testing the models of CNN,CNN-RNNs (LSTM,GRU,BiLSTM,BiGRU), and CNNTransformer
3.	See Intra_Train_CNNs.ipynb","Intra_Train_CNNRNNs.ipynb","Intra_Train_CNNTransformer.ipynb" files in "trainTest" folder to conduct model training and testing.
4.	See "Intra_Train_MLs.ipynb" file in "trainTest" folderfor the comparison with the state-of-art Machine Learing(ML) models.



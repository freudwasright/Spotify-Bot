import cv2
import os

DataFilePath = os.listdir('D:/Data/images_original')
DataFilePath.sort()
for CategoryName in DataFilePath:
    CategoryFolder = os.listdir(CategoryName)
    CategoryFolder.sort()
    for MelSpectrogram in CategoryFolder:
        CurrentImage = cv2.imread(MelSpectrogram)

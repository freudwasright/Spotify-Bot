# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cv2


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.

    img = cv2.imread('D:/Data/images_original/blues/blues00000.png')
    print(img.shape)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

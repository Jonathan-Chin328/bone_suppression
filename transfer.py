import cv2
import matplotlib as plt
import glob

'''
Transform image in present folder
1. black <--> white
2. equalize histogram
'''

src = sorted(glob.glob('./dataset/augmented/convert_augmented/source/*'))
tgt = sorted(glob.glob('./dataset/augmented/convert_augmented/target/*'))

def convert(img_list):
    n = 0
    for fname in img_list:
        n += 1
        if n % 1000 == 0:
            print(n)
        img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        convert_img = 255 - img
        equ = cv2.equalizeHist(convert_img)
        equ[equ[:,:] == 255] = 0
        cv2.imwrite(fname, equ)

# convert(src)
# convert(tgt)
import os
import glob
import cv2
import numpy as np
import json

from sklearn.decomposition import PCA
from tqdm import tqdm

src = sorted(glob.glob(os.path.join('./dataset/augmented/convert_augmented', 'source/*')))
pca_source = []
for fname in tqdm(src):
    src_img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    src_img = src_img / 255
    pca_horizon = PCA(n_components=256)
    pca_vertical = PCA(n_components=256)
    image_pca = pca_horizon.fit_transform(src_img)
    image_pca2 = pca_vertical.fit_transform(image_pca.T)
    src_img = np.diag(image_pca2)
    pca_source.append(src_img.tolist())

data = {
    'pca_source': pca_source
}

with open("dataset.json", 'w') as file:
    file.write(json.dumps(data))
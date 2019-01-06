from keras.applications import InceptionV3
from keras.layers import GlobalAveragePooling2D
from keras.models import Model

from PIL import Image, ImageDraw
from sklearn.manifold import TSNE
import joblib
import numpy as np
from tqdm import tqdm
import cv2

def create_net():
    inception = InceptionV3(include_top=False, weights="imagenet", input_shape=(299,299,3))
    x = GlobalAveragePooling2D()(inception.output)

    return Model(inception.inputs, x)

def generator(batch_size):
    metadata = joblib.load("metadata.job.xz")
    X_cache = []
    while True:
        for i, path in metadata.items():
            with Image.open(path) as img:
                width, height = img.size
                img = img.convert("RGB")
                left = (width - height) // 2
                right = left + height

                img = img.crop((left, 0, right, height)).resize((299,299), Image.BILINEAR)
                img_array = np.asarray(img, dtype=np.uint8)
                X_cache.append(img_array)

                if len(X_cache) == batch_size:
                    X_batch = (np.asarray(X_cache) / 255.0).astype(np.float32)
                    X_cache = []
                    yield X_batch
        
def latent_values():
    model = create_net()
    # 23135 = 35 * 661
    latent = model.predict_generator(generator(35), steps=661, max_queue_size=1, verbose=1)
    joblib.dump(latent, "latent.job.xz", compress=3)

def embedding():
    data = joblib.load("latent.job.xz")
    tsne = TSNE(n_components=2, perplexity=10.0)
    embedded = tsne.fit_transform(data)
    print(embedded.shape)
    joblib.dump(embedded, "tsne.job.xz", compress=3)

import matplotlib.pyplot as plt

def plot_embeddiing():
    data = joblib.load("tsne.job.xz")
    plt.scatter(data[:,0], data[:,1], marker=".")
    plt.show()

def find_nearest_neighbor():
    # 60x30でプロット
    data = joblib.load("tsne.job.xz")
    min_x = min(data[:, 0])
    max_x = max(data[:, 0])
    min_y = min(data[:, 1])
    max_y = max(data[:, 1])
    pitch_x = (max_x - min_x) / 60.0
    pitch_y = (max_y - min_y) / 30.0
    index_mat = np.zeros((60, 30), dtype=np.int32)
    for i in tqdm(range(60)):
        for j in tqdm(range(30)):
            pos_x = min_x + pitch_x * i
            pos_y = min_y + pitch_y * j
            positions = np.array([pos_x, pos_y]).reshape(1, -1)
            dist = np.sum((data - positions) ** 2, axis=-1)
            index = np.argmin(dist)
            index_mat[i, j] = index
    joblib.dump(index_mat, "indices.job.xz", compress=3)

def paste_images(file_path, is_upper_triangle, target_image, position):
    with Image.new("L", (128, 128)) as mask:
        draw = ImageDraw.Draw(mask)
        if is_upper_triangle:
            draw.polygon([(64, 0), (0, 128), (128, 128)], fill=255)
        else:
            draw.polygon([(0, 0), (128, 0), (64, 128)], fill=255)
        with Image.open(file_path) as original:
            width, height = original.size
            left = (width - height) // 2
            right = left + height

            resized = original.crop((left, 0, right, height)).resize((128,128), Image.BILINEAR).convert("RGBA")
            resized.putalpha(mask)
            target_image.paste(resized, position, resized)

def merge_images():
    metadata = joblib.load("metadata.job.xz")
    neighbors = joblib.load("indices.job.xz")
    with Image.new("RGBA", (64*60, 128*30)) as canvas:
        for i in tqdm(range(neighbors.shape[0])):
            for j in tqdm(range(neighbors.shape[1])):
                paste_images(metadata[neighbors[i, j]], (i+j)%2==0, canvas, (-64+64*i, 128*j))
        canvas.save("merged.png")
    # 1分半ぐらいかかる
  
if __name__ == "__main__":
    merge_images()

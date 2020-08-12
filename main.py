import argparse
import os
import imageio
from PIL import Image
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from create_model import create_model


def model_work(model, filename, images_dir):
    file_for_prnt = os.path.basename(filename)
    img = Image.open(filename)
    width, height = img.size
    img.close()

    if width > 28 or height > 28:
        img = image.load_img(filename, target_size=(28, 28), color_mode='grayscale')
        fname = os.path.join(images_dir, f'resize_{file_for_prnt}')
        # в задаче особо четких указаний не было, поэтому решил их просто оставлять после обработки
        img.save(fname)
        filename = fname

    img = imageio.imread(filename, as_gray=False, pilmode="RGB")
    img = np.mean(img, 2, dtype=float)
    img = img / 255
    img = np.expand_dims(img, 0)
    img = np.expand_dims(img, -1)

    print(f'file "{file_for_prnt}" = {np.argmax(model.predict(img))}')


if __name__ == '__main__':
    parse_m = argparse.ArgumentParser(description='some description')
    parse_m.add_argument('-dirImages', '--dirImages', type=str)
    args = parse_m.parse_args()

    if not os.path.exists(os.path.join(os.path.abspath(os.curdir), 'my_model.h5')):
        create_model()
    model = keras.models.load_model('my_model.h5')

    my_suffixes = ("jpeg", "png", "jpg")
    images_dir = os.path.join(os.path.abspath(os.curdir), 'images')

    if args.dirImages:
        images_dir = args.dirImages

    for file in os.listdir(images_dir):
        if file.endswith(my_suffixes):
            model_work(model, filename=os.path.join(images_dir, file), images_dir=images_dir)


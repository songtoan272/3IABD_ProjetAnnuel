#!/usr/bin/env python

import sys
from PIL import Image
import os
from os import listdir
from os.path import isfile, join

def get_all_images(path_images):
    result = []

    images = [f for f in listdir(path_images) if isfile(join(path_images, f))]
    for img in images:
        full_path = path_images + '/' + img
        directory = os.path.dirname(os.path.realpath(full_path))
        name = os.path.splitext(os.path.basename(full_path))[0]
        extension = os.path.splitext(os.path.basename(full_path))[1]
        result.append([full_path, directory, name, extension])
    return result


if __name__ == "__main__":
    resize_several = False
    conserve_ratio = False
    crop_image = True

    width = 16
    height = width

    images = []
    fill_color = (0, 0, 0, 0)

    if resize_several:
        images = get_all_images("D:/Utilisateurs/Bureau/projet_annuel/images_to_resize")
    else:
        path = sys.argv[1]
        width = int(sys.argv[2])
        height = int(sys.argv[3])
        ratio = int(sys.argv[4])

        if ratio == 1:
            conserve_ratio = True
            crop_image = False

        directory = os.path.dirname(os.path.realpath(path))
        name = os.path.splitext(os.path.basename(path))[0]
        extension = os.path.splitext(os.path.basename(path))[1]
        images.append([path, directory, name, extension])

    for img in images:
        if resize_several:
            new_file = img[1] + '/resized/' + img[2] + '.png'
        else:
            new_file = img[1] + '/' + img[2] + '_resized.png'
        with open(img[0], 'r+b') as f:
            with Image.open(f) as image:
                img_width, img_height = image.size

                if crop_image:
                    side = min(img_width, img_height)
                    left = (img_width - side) / 2
                    right = (img_width + side) / 2
                    top = 0
                    bottom = side
                    image = image.crop((left, top, right, bottom))

                if conserve_ratio:
                    n_im = Image.new('RGBA', (width, height), fill_color)
                    image.thumbnail((width, height), Image.ANTIALIAS)
                    x, y = image.size
                    n_im.paste(image, (int((width - x) / 2), int((height - y) / 2)))
                else:
                    n_im = image.resize((width, height))
                n_im.save(new_file, 'PNG')
        print(new_file)

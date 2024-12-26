import os
import os.path as osp

import numpy as np
import tqdm
from osgeo import gdal

from fastimg_func import IMAGE4, pad_win, depad_win


def read_tiff(image_path):
    ds = gdal.Open(image_path)
    return ds.ReadAsArray()


def crop_pad_extent(img_path, dst_dir, winsize, padding, ignore_zero=True):
    os.makedirs(dst_dir, exist_ok=True)
    img = IMAGE4(img_path)
    exts = img.gen_extents(winsize, winsize)
    for idx, ext in enumerate(tqdm.tqdm(exts)):
        dst_path = osp.join(dst_dir, f'{idx}.tif')
        pad_data = pad_win(img, ext, padding=padding)
        if ignore_zero:
            if np.all(pad_data == 0):
                continue
        w, h = pad_data.shape[1:]
        img.create_img(dst_path, im_width=w, im_height=h, im_geotrans=None, datatype=gdal.GDT_Byte)
        img.write_extent(im_data=pad_data)


def pad_merge_extent(img_path, ext_dir, out_path):
    img = IMAGE4(img_path)
    exts = img.gen_extents(winsize, winsize)
    img.create_img(out_path)
    for idx, ext in enumerate(exts):
        ext_path = osp.join(ext_dir, f'{idx}.tif')
        if osp.exists(ext_path):
            pad_data = read_tiff(ext_path)
            depad_data = depad_win(pad_data, padding=padding)
            img.write_extent(depad_data, ext)


if __name__ == '__main__':
    img_path = r"D:\Desktop\tmp\mos.tif"
    dst_dir = r'D:\Desktop\tmp\ext'
    winsize = 1024
    padding = 50
    crop_pad_extent(img_path, dst_dir, winsize, padding)
    pad_merge_extent(img_path, dst_dir, osp.join(dst_dir, '0000.tif'))

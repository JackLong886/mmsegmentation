import os
import secrets
import sys

import numpy as np
import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from fastimg_func import IMAGE3, coord_ras2geo, coord_geo2ras
from osgeo import gdal

if __name__ == '__main__':
    image_list = [
        "D:\LCJ\datasets\sugarcane\label.tif",
        "D:\LCJ\datasets\sugarcane\clip.tif",
    ]
    dst_dir_list = [
        r"D:\LCJ\datasets\sugarcane\label",
        r"D:\LCJ\datasets\sugarcane\sugar_ext"
    ]
    for dst_dir in dst_dir_list:
        os.makedirs(dst_dir, exist_ok=True)
    winsize_list = [2048, 1024, 512]
    for winsize in winsize_list:
        for i in range(len(image_list) // 2):
            print(image_list[i * 2])
            print(image_list[i * 2 + 1])
            img1 = IMAGE3(image_list[i * 2])
            img2 = IMAGE3(image_list[i * 2 + 1])
            exts = img1.gen_extents(winsize, winsize)
            for idx, ext in enumerate(tqdm.tqdm(exts)):
                imdata1 = img1.get_extent(ext)
                if np.all(imdata1 == 0):
                    continue
                name = secrets.token_hex(8)
                geo_lu = coord_ras2geo(img1.im_geotrans, [ext[0], ext[1]])
                geotrans1 = (
                    geo_lu[0], img1.im_geotrans[1], img1.im_geotrans[2], geo_lu[1], img1.im_geotrans[4],
                    img1.im_geotrans[5])
                dst_path1 = os.path.join(dst_dir_list[i * 2], f'{name}.tif')

                c, r = coord_geo2ras(img2.im_geotrans, [geo_lu[0], geo_lu[1]])
                ext = [c, r, winsize, winsize]
                geotrans2 = (
                    geo_lu[0], img2.im_geotrans[1], img2.im_geotrans[2], geo_lu[1], img2.im_geotrans[4],
                    img2.im_geotrans[5])
                dst_path2 = os.path.join(dst_dir_list[i * 2 + 1], f'{name}.tif')

                imdata2 = img2.get_extent(ext)
                img1.create_img(dst_path1, im_width=winsize, im_height=winsize, im_geotrans=geotrans1,
                                datatype=gdal.GDT_Byte)
                img1.write_extent(im_data=imdata1)
                img2.create_img(dst_path2, im_width=ext[2], im_height=ext[3], im_geotrans=geotrans2,
                                datatype=gdal.GDT_Byte)

                img2.write_extent(im_data=imdata2)

import io
import json
import os
import tarfile
from glob import glob

import numpy as np
from osgeo import gdal, osr
import torch.jit


def find_dir_list(path):
    if os.path.exists(path):
        path_list = []
        for path, dir_lst, _ in os.walk(path):
            for dir_name in dir_lst:
                path_list.append(os.path.join(path, dir_name))
        return path_list


def get_json2opt(json_path, opt):
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8-sig') as f:
            js = json.load(f)
            for key in js:
                if key in vars(opt).keys():
                    vars(opt)[key] = js[key]
    return opt


def un_tar(un_tar_path, tar_path=None, tar_path_list=None):
    if tar_path:
        tar_path_list = glob(tar_path + r'\*.tar.gz')
    for path in tar_path_list:
        print('解压文件{}'.format(path))
        tf = tarfile.open(path, 'r')
        preffix, _ = os.path.splitext(os.path.basename(path))
        preffix, _ = os.path.splitext(os.path.basename(preffix))
        un_tar_path2 = os.path.join(un_tar_path, preffix)
        tf.extractall(path=un_tar_path2)
        tf.close()
    return un_tar_path


def make_file(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_file_size_in_mb(file_path):
    """计算文件大小并转换为MB"""
    if os.path.isfile(file_path):
        size_bytes = os.path.getsize(file_path)
        size_mb = size_bytes / (1024 * 1024) + 1
        return int(round(size_mb, 2))
    else:
        return "File not found."


def get_filelist(dir, Filelist):
    if os.path.isfile(dir):
        Filelist.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            # 如果需要忽略某些文件夹，使用以下代码
            # if s == "xxx":
            # continue
            newDir = os.path.join(dir, s)
            get_filelist(newDir, Filelist)


def torch_load_chinese(file_path, device):
    with open(file_path, 'rb') as f:
        buffer = io.BytesIO(f.read())
    return torch.jit.load(buffer, map_location=device)


def frmt2suffix(frmt):
    if frmt == 'GTiff':
        suffix = '.tif'
    elif frmt == 'HFA':
        suffix = '.img'
    elif frmt == 'VRT':
        suffix = '.vrt'
    else:
        raise KeyError('{}不是支持的文件格式'.format(frmt))
    return suffix


def path2frmt(img_path):
    suffix = os.path.splitext(os.path.basename(img_path))[1]
    if suffix in ['.tif', '.TIF', '.tiff', '.TIFF']:
        frmt = 'GTiff'
    elif suffix in ['.img', '.IMG']:
        frmt = 'HFA'
    elif suffix in ['.vrt', '.VRT']:
        frmt = 'VRT'
    else:
        raise KeyError('{}不是支持的文件格式'.format(suffix))
    return frmt


def read_txt_file(txt_path):
    content_list = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip('\n')  # 去掉列表中每一个元素的换行符
            content_list.append(line.replace("'", "").replace('"', "").replace(",", ""))
    return content_list


def update_json(json_file_path, info_dict):
    if not os.path.exists(json_file_path):
        with open(json_file_path, 'w') as json_file:
            json.dump(info_dict, json_file, indent=2)
    else:
        with open(json_file_path, 'r') as json_file:
            json_data = json.load(json_file)
        for key, val in info_dict.items():
            json_data[key] = val
        with open(json_file_path, 'w') as json_file:
            json.dump(json_data, json_file, indent=2)


def check_shp(shp_fileOrlist):
    if not shp_fileOrlist:
        raise AssertionError('请输入待检查影像')
    if not isinstance(shp_fileOrlist, list):
        shp_list = [shp_fileOrlist]
    else:
        shp_list = shp_fileOrlist
    for shp_file in shp_list:
        if not os.path.exists(shp_file):
            raise ValueError(f'矢量不存在：{shp_file}')
        else:
            # todo 检查格式
            pass


def check_img(img_pathOrlist, **kwargs):
    """
    检查是否影像存在、投影、3波段, 是否存在有效值
    ['bands', 'proj', 'bit_depth', 'valid_data']
    """
    if not img_pathOrlist:
        raise AssertionError('请输入待检查影像')
    if not isinstance(img_pathOrlist, list):
        img_list = [img_pathOrlist]
    else:
        img_list = img_pathOrlist
    for img_path in img_list:
        if not os.path.exists(img_path):
            raise AssertionError(f'影像不存在：{img_path}')
        else:
            ds = gdal.Open(img_path)
            bands = ds.RasterCount
            if 'bands' in kwargs:
                if bands != kwargs['bands']:
                    raise AssertionError(f"影像波段为:{bands}, 需输入{kwargs['bands']}波段影像 \n{img_path}")
            if 'proj' in kwargs:
                im_proj = ds.GetProjection()
                proj_srs = osr.SpatialReference()
                proj_srs.ImportFromWkt(im_proj)
                if not proj_srs.IsProjected():
                    raise AssertionError(f'影像无投影： {img_path}')
            if 'bit_depth' or 'valid_data' in kwargs:
                stats = []
                max_v = -1
                for i in range(bands):
                    s = ds.GetRasterBand(i + 1).ComputeStatistics(approx_ok=True)
                    stats.append(s)
                    max_v = max(max_v, s[1])
                if 'bit_depth' in kwargs:
                    bit_depth = int(np.log2(max_v) + 1)
                    if bit_depth > kwargs['bit_depth']:
                        raise AssertionError(f"影像深度为{bit_depth}, 需输入深度为{kwargs['bit_depth']}： \n{img_path}")
                if 'valid_data' in kwargs:
                    for s in stats:
                        if s[2] < kwargs['valid_data']:
                            raise AssertionError(f"请检查影像像素有效值：\n{img_path}")
            ds = None


if __name__ == '__main__':
    img_list = [
        r"D:\LCJ\DOM\ZY1E_VNIC_E110.4_N24.2_20221224_L1B0000546824-PAN_output.tif",
        r"D:\LCJ\DOM\GF2_PMS2_E110.3_N24.5_20220928_L1A0006781153-PAN2_output.tif",
        r"D:\LCJ\DOM\GF1B_PMS_E110.2_N24.6_20221017_L1A1228202900-PAN_output.tif",
        r"D:\LCJ\DOM\GF1_PMS1_E110.2_N24.4_20221026_L1A0006852535-PAN1_output.tif",
        r"D:\LCJ\DOM\1_Mosaic.tif",
        r"D:\LCJ\DOM\rem_ZY1E_VNIC_E110.4_N24.2_20221224_L1B0000546824-PAN_output.tif",
        r"D:\LCJ\DOM\rem_GF1_PMS1_E110.2_N24.4_20221026_L1A0006852535-PAN1_output.tif",
    ]
    check_img(img_list, bands=3, bit_depth=8, valid_data=20)

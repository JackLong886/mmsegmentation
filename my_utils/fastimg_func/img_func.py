import datetime
import math
import os
import re
import secrets

import cv2
import numpy as np
from numpy import einsum
from osgeo import gdal, osr, ogr

from . import frmt2suffix, get_file_size_in_mb
from .coord_func import coord_geo2ras, proj2geo
from .file_func import make_file
from .reproj import shp2img_reproj


class IMAGE3:
    def __init__(self, filenameOrDs, read_type=gdal.GA_ReadOnly, if_print=False):
        """
        Args:
            filenameOrDs: filename or datasets
            read_type: 0-gdal.GA_ReadOnly, 1-gdal.GA_Update
            if_print:
        """
        self.output_dataset = None
        self.copy_dataset = None
        self.dataset = None
        self.read_type = read_type
        self.bit_depth = None
        self.statis = None
        self.in_file = None
        self.copy_image_file = None
        if isinstance(filenameOrDs, gdal.Dataset):
            self.dataset = filenameOrDs
        elif os.path.isfile(filenameOrDs):
            self.in_file = filenameOrDs
            self.dataset = gdal.Open(self.in_file, read_type)  # 打开文件
        else:
            raise KeyError('无法通过 {} 初始化IMAGE'.format(filenameOrDs))
        self.im_width = self.dataset.RasterXSize  # 栅格矩阵的列数
        self.im_height = self.dataset.RasterYSize  # 栅格矩阵的行数
        self.im_bands = self.dataset.RasterCount  # 波段数
        self.im_geotrans = self.dataset.GetGeoTransform()  # 仿射矩阵，左上角像素的大地坐标和像素分辨率
        self.im_srs = self.dataset.GetSpatialRef()
        self.im_proj = self.dataset.GetProjection()  # 地图投影信息，字符串表示
        self.im_resx, self.im_resy = self.im_geotrans[1], self.im_geotrans[5]
        self.datatype = self.dataset.GetRasterBand(1).DataType
        try:
            self.proj_srs = osr.SpatialReference()
            self.proj_srs.ImportFromWkt(self.im_proj)
            self.geo_srs = self.proj_srs.CloneGeogCS()
        except:
            pass

        if if_print:
            print('*' * 200)
            print("in_file:{}".format(self.in_file))
            print("res_x:{}, res_y:{}".format(self.im_resx, self.im_resy))
            print("width:{}, height:{}, bands:{}".format(self.im_width, self.im_height, self.im_bands))
            print("im_geotrans:{}".format(self.im_geotrans))
            print("im_proj:{}".format(self.im_proj))
            print('*' * 200)

    def get_extent(self, extent=None, bands=None):
        if extent is None:
            x, y, x_size, y_size = 0, 0, self.im_width, self.im_height
        else:
            x, y, x_size, y_size = extent

        if x + x_size > self.im_width:
            x = self.im_width - x_size
        if y + y_size > self.im_height:
            y = self.im_height - y_size

        if bands is None:
            return self.dataset.ReadAsArray(x, y, x_size, y_size)
        if isinstance(bands, int):
            assert bands <= self.im_bands
            return self.dataset.GetRasterBand(bands + 1).ReadAsArray(x, y, x_size, y_size)
        elif isinstance(bands, list):
            out_data = []
            for band in bands:
                assert band <= self.im_bands
                out_data.append(self.dataset.GetRasterBand(band + 1).ReadAsArray(x, y, x_size, y_size))
            return np.stack(out_data, axis=0)

    def write2self_img(self, im_data=None, extent=None):
        assert self.read_type == gdal.GA_Update
        if not extent:
            extent = (0, 0, self.im_width, self.im_height)

        if self.im_bands == 1:
            self.dataset.GetRasterBand(1).WriteArray(im_data, xoff=extent[0], yoff=extent[1])
        else:
            for i in range(self.im_bands):
                self.dataset.GetRasterBand(i + 1).WriteArray(im_data[i], xoff=extent[0], yoff=extent[1])
        self.dataset.FlushCache()

    def create_img(self, filename, out_bands=None, im_width=None, im_height=None, im_proj=None, im_geotrans=None,
                   datatype=None, block_size=(256, 256), nodata=0, im_srs=None):
        """
            datatype: 1-gdal.GDT_Byte, 2-gdal.GDT_UInt16, 6-gdal.GDT_Float32
        """
        self.output_file = filename
        # 创建文件
        driver = gdal.GetDriverByName(self.get_frmt(filename))
        options = ['TILED=YES', 'BLOCKXSIZE={}'.format(block_size[0]), 'BLOCKYSIZE={}'.format(block_size[1])]
        if not datatype:
            datatype = self.datatype

        self.out_bands = self.im_bands if not out_bands else out_bands
        if im_width and im_height:
            self.output_dataset = driver.Create(filename, im_width, im_height, self.out_bands, datatype,
                                                options=options)
        else:
            self.output_dataset = driver.Create(filename, self.im_width, self.im_height, self.out_bands, datatype,
                                                options=options)
        if im_geotrans:
            self.output_dataset.SetGeoTransform(im_geotrans)
        else:
            self.output_dataset.SetGeoTransform(self.im_geotrans)  # 写入仿射变换参数
        if im_proj:
            self.output_dataset.SetProjection(im_proj)
        else:
            self.output_dataset.SetProjection(self.im_proj)  # 写入投影

        if im_srs:
            self.output_dataset.SetSpatialRef(im_srs)
        else:
            self.output_dataset.SetSpatialRef(self.im_srs)

        if nodata is not None:
            for band in range(self.out_bands):
                self.output_dataset.GetRasterBand(band + 1).SetNoDataValue(nodata)
        self.output_dataset.FlushCache()

    def write_extent(self, im_data=None, extent=None, bands=None):
        if im_data is None: return 0
        (x, y, x_size, y_size) = extent if extent is not None else (0, 0, self.im_width, self.im_height)

        if x + x_size > self.im_width:
            x = self.im_width - x_size
        if y + y_size > self.im_height:
            y = self.im_height - y_size

        if self.out_bands == 1:
            assert len(im_data.shape) == 2
            self.output_dataset.GetRasterBand(1).WriteArray(im_data, xoff=x, yoff=y)  # 写入数组数据
        else:
            if bands is not None:
                if isinstance(bands, int):
                    assert bands <= self.out_bands
                    self.output_dataset.GetRasterBand(bands + 1).WriteArray(im_data, xoff=x, yoff=y)
                elif isinstance(bands, list):
                    for i, band in enumerate(bands):
                        assert band <= self.im_bands
                        self.output_dataset.GetRasterBand(band + 1).WriteArray(im_data[i], xoff=x, yoff=y)
            else:
                for i in range(self.out_bands):
                    self.output_dataset.GetRasterBand(i + 1).WriteArray(im_data[i], xoff=x, yoff=y)
        self.output_dataset.FlushCache()

    def compute_statistics(self, if_print=False, approx_ok=True):
        # min max mean std
        statis = []
        for i in range(self.im_bands):
            s = self.dataset.GetRasterBand(i + 1).ComputeStatistics(approx_ok)
            statis.append(s)
        if if_print:
            for i in range(len(statis)):
                print("min:{}, max:{}, mean:{}, std:{}".format(*statis[i]), flush=True)
        self.statis = statis
        return statis

    def compute_bit_depth(self, approx_ok=True):
        if not self.statis:
            self.compute_statistics(approx_ok=approx_ok)
        bit_depth = 0
        for s in self.statis:
            bit_depth = max(bit_depth, 2 ** math.ceil(math.log2(s[1])))

        self.bit_depth = bit_depth
        return bit_depth

    def compute_bit_depth_lst(self, approx_ok=True):
        if not self.statis:
            self.compute_statistics(approx_ok=approx_ok)
        self.bit_depth_lst = []
        for s in self.statis:
            self.bit_depth_lst.append(2 ** math.ceil(math.log2(s[1])))
        return self.bit_depth_lst

    def gen_extents(self, x_winsize, y_winsize, win_std=None):
        if win_std is None:
            win_std = [x_winsize, y_winsize]
        frame = []
        x = 0
        y = 0
        while y < self.im_height:  # 高度方向滑窗
            if y + y_winsize >= self.im_height:
                y_left = self.im_height - y_winsize
                y_right = y_winsize
                y_end = True
            else:
                y_left = y
                y_right = y_winsize
                y_end = False

            while x < self.im_width:  # 宽度方向滑窗
                if x + x_winsize >= self.im_width:
                    x_left = self.im_width - x_winsize
                    x_right = x_winsize
                    x_end = True
                else:
                    x_left = x
                    x_right = x_winsize
                    x_end = False
                frame.append((x_left, y_left, x_right, y_right))
                x += win_std[0]
                if x_end:
                    break
            y += win_std[1]
            x = 0
            if y_end:
                break
        return frame

    def __del__(self):
        self.dataset = None
        self.copy_dataset = None
        self.output_dataset = None

    def copy_image(self, filename, delete=False):
        print(f'开始文件复制{filename}')
        self.copy_image_file = filename
        if os.path.exists(filename) and delete is False:
            pass
        else:
            if os.path.exists(filename):
                os.remove(filename)
            dirname = os.path.dirname(filename)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            # 同类型间拷贝最快
            if not self.in_file:
                gdal.Translate(self.copy_image_file, self.dataset, format=self.get_frmt(self.copy_image_file))
            else:
                if os.path.splitext(os.path.basename(self.copy_image_file))[1] == \
                        os.path.splitext(os.path.basename(self.in_file))[1]:
                    gdal.GetDriverByName(self.get_frmt(self.copy_image_file)).CopyFiles(self.copy_image_file,
                                                                                        self.in_file)
                else:
                    gdal.Translate(self.copy_image_file, self.in_file, format=self.get_frmt(self.copy_image_file))
            os.chmod(self.copy_image_file, 0o755)  # 设置目标文件的权限

        self.copy_dataset = gdal.Open(self.copy_image_file, gdal.GA_Update)
        self.copy_dataset.BuildOverviews("NONE", [])
        if self.copy_dataset:
            print("文件复制成功！")
        else:
            raise KeyError("文件复制失败:{}".format(filename))

    def write2copy_image(self, extent, im_data):
        x, y, _, _ = extent
        bands = self.copy_dataset.RasterCount
        if bands == 1:
            self.copy_dataset.GetRasterBand(1).WriteArray(im_data, xoff=x, yoff=y)  # 写入数组数据
        else:
            for i in range(bands):
                self.copy_dataset.GetRasterBand(i + 1).WriteArray(im_data[i], xoff=x, yoff=y)
        self.copy_dataset.FlushCache()

    def get_4_extent(self, dataset=None):
        if not dataset:
            # 计算影像的四至范围
            x_min = self.im_geotrans[0]
            y_max = self.im_geotrans[3]
            x_max = x_min + self.im_geotrans[1] * self.im_width
            y_min = y_max + self.im_geotrans[5] * self.im_height
        else:
            # 获取影像的地理转换
            geotransform = dataset.GetGeoTransform()
            # 获取影像的宽度和高度
            width = dataset.RasterXSize
            height = dataset.RasterYSize
            # 计算影像的四至范围
            x_min = geotransform[0]
            y_max = geotransform[3]
            x_max = x_min + geotransform[1] * width
            y_min = y_max + geotransform[5] * height
        return x_min, y_min, x_max, y_max

    def get_4ext_geom(self):
        dataset = self.dataset
        minx, miny, maxx, maxy = self.get_4_extent(dataset)
        minx, miny = proj2geo(self.im_proj, minx, miny)
        maxx, maxy = proj2geo(self.im_proj, maxx, maxy)

        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(minx, miny)
        ring.AddPoint(maxx, miny)
        ring.AddPoint(maxx, maxy)
        ring.AddPoint(minx, maxy)
        ring.AddPoint(minx, miny)
        bbox_geom = ogr.Geometry(ogr.wkbPolygon)
        bbox_geom.AddGeometry(ring)
        return bbox_geom

    def get_frmt(self, img_path):
        self.suffix = os.path.splitext(os.path.basename(img_path))[1]
        if self.suffix in ['.tif', '.TIF', '.tiff', '.TIFF']:
            self.frmt = 'GTiff'
        elif self.suffix in ['.img', '.IMG']:
            self.frmt = 'HFA'
        elif self.suffix in ['.vrt', '.VRT']:
            self.frmt = 'VRT'
        else:
            raise KeyError('{}不是支持的文件格式'.format(self.suffix))
        return self.frmt

    def is_projected(self):
        proj_srs = osr.SpatialReference()
        proj_srs.ImportFromWkt(self.im_proj)
        return proj_srs.IsProjected()

    def ov_rgb_histogram(self, level):
        overview_count = self.dataset.GetRasterBand(1).GetOverviewCount()
        if not overview_count:
            # 清理原来的金字塔
            self.dataset.BuildOverviews("NONE", [])
            # 新建指定的金字塔
            self.dataset.BuildOverviews("NEAREST", [level])
        self.compute_bit_depth()
        spline_list = []
        low_percent = 2
        high_percent = 98
        gain = 1
        # 循环统计
        for i in range(3):
            band = self.dataset.GetRasterBand(3 - i)
            ov = band.GetOverview(0)
            hist = ov.GetHistogram(min=1,
                                   max=self.bit_depth - 1,
                                   buckets=self.bit_depth - 1,
                                   include_out_of_range=0,
                                   approx_ok=0)
            hist = np.array(hist)
            # hist[hist == 0] = 1
            total = np.sum(hist)
            cumulative_percentages = np.cumsum(hist) / total * 100

            low_value = np.argmax(cumulative_percentages >= low_percent)
            high_value = np.argmax(cumulative_percentages >= high_percent)

            low_percent = cumulative_percentages[low_value + 1]
            high_percent = cumulative_percentages[high_value - 1]

            min_percent = low_percent + 2.5
            max_percent = high_percent - 2.5

            min_value = np.argmax(cumulative_percentages >= min_percent)
            max_value = np.argmax(cumulative_percentages >= max_percent)

            if np.mean(hist[:low_value]) > hist[low_value]:
                low_value = low_value + np.argmin(hist[low_value:min_value])
                low_percent = cumulative_percentages[low_value]
                min_percent = low_percent + 2.5
                max_percent = high_percent - 2.5
                min_value = np.argmax(cumulative_percentages >= min_percent)
                max_value = np.argmax(cumulative_percentages >= max_percent)
            if np.mean(hist[high_value:]) > hist[high_value]:
                high_value = max_value + np.argmin(hist[max_value:high_value])
                high_percent = cumulative_percentages[high_value]
                min_percent = low_percent + 2.5
                max_percent = high_percent - 2.5
                min_value = np.argmax(cumulative_percentages >= min_percent)
                max_value = np.argmax(cumulative_percentages >= max_percent)

            high_freq = np.max(hist[low_value:high_value])
            max_freq = hist[max_value]
            min_freq = hist[min_value]

            max_range = hist[max_value:high_value]
            min_range = hist[low_value:min_value]

            max_freq_mean = np.mean(max_range[max_range != 0])
            min_freq_mean = np.mean(min_range[min_range != 0])

            min_ratio = min_freq / min_freq_mean
            max_ratio = max_freq / max_freq_mean

            min_end = 2 * min_percent - low_percent
            max_end = 2 * max_percent - max_percent

            if i < 3:
                if min_ratio > high_freq / min_freq:
                    while True:
                        if min_percent < low_percent:
                            min_percent = low_percent
                            min_value = np.argmax(cumulative_percentages >= min_percent)
                            break
                        min_percent -= 0.05
                        min_value = np.argmax(cumulative_percentages >= min_percent)
                        min_freq = hist[min_value]
                        min_range = hist[low_value:min_value]
                        min_freq_mean = np.mean(min_range[min_range != 0])
                        ratio_min = min_freq / min_freq_mean
                        if ratio_min < high_freq / min_freq:
                            low_percent = min_percent
                            break
                else:
                    while True:
                        if min_percent > min_end:
                            min_percent = min_end
                            min_value = np.argmax(cumulative_percentages >= min_percent)
                            break
                        min_percent += 0.05
                        min_value = np.argmax(cumulative_percentages >= min_percent)
                        min_freq = hist[min_value]
                        min_range = hist[low_value:min_value]
                        min_freq_mean = np.mean(min_range[min_range != 0])
                        ratio_min = min_freq / min_freq_mean
                        if ratio_min > high_freq / min_freq:
                            break

                if max_ratio > high_freq / max_freq:
                    while True:
                        if max_percent > high_percent:
                            max_percent = high_percent
                            max_value = np.argmax(cumulative_percentages >= max_percent)
                            break
                        max_percent += 0.05
                        max_value = np.argmax(cumulative_percentages >= max_percent)
                        max_freq = hist[max_value]
                        max_range = hist[max_value:high_value]
                        max_freq_mean = np.mean(max_range[max_range != 0])
                        ratio_max = max_freq / max_freq_mean
                        if ratio_max < high_freq / max_freq:
                            break
                else:
                    while True:
                        if max_percent < max_end:
                            max_percent = max_end
                            max_value = np.argmax(cumulative_percentages >= max_percent)
                            break
                        max_percent -= 0.05
                        max_value = np.argmax(cumulative_percentages >= max_percent)
                        max_freq = hist[max_value]
                        max_range = hist[max_value:high_value]
                        max_freq_mean = np.mean(max_range[max_range != 0])
                        ratio_max = max_freq / max_freq_mean
                        if ratio_max > high_freq / max_freq:
                            break
                if i == 0:
                    gain = self.bit_depth / high_value
                    gain = min(np.e, gain)
            spline_list.append(
                [0,
                 int(min_value),
                 int(max_value),
                 int(max_value * gain)
                 ])
        return spline_list


def pad_win(img, extent, padding):
    # 解析和调整区域坐标
    col_off, row_off, local_width, local_height = extent
    col_off -= padding
    row_off -= padding
    local_width += 2 * padding
    local_height += 2 * padding

    # 获取图像的全局属性
    bands, global_width, global_height = img.im_bands, img.im_width, img.im_height

    # 初始化偏移量
    left_offset = max(0, -col_off)
    top_offset = max(0, -row_off)
    right_offset = max(0, col_off + local_width - global_width)
    bottom_offset = max(0, row_off + local_height - global_height)

    # 计算新的边界值
    c = max(0, col_off)
    r = max(0, row_off)
    w = min(local_width - left_offset - right_offset, global_width - c)
    h = min(local_height - top_offset - bottom_offset, global_height - r)

    # 获取图像数据并初始化输出数组
    imdata = img.get_extent([c, r, w, h])
    if len(imdata.shape) == 2:
        imdata = imdata[np.newaxis, :, :]
    out = np.zeros((bands, local_height, local_width), dtype=np.float32)
    out[:, top_offset:top_offset + h, left_offset:left_offset + w] = imdata
    # 镜像填充
    if left_offset > 0:
        out[:, :, :left_offset] = np.flip(out[:, :, left_offset:2 * left_offset], axis=2)
    if top_offset > 0:
        out[:, :top_offset, :] = np.flip(out[:, top_offset:2 * top_offset, :], axis=1)
    if right_offset > 0:
        out[:, :, -right_offset:] = np.flip(out[:, :, -2 * right_offset:-right_offset], axis=2)
    if bottom_offset > 0:
        out[:, -bottom_offset:, :] = np.flip(out[:, -2 * bottom_offset:-bottom_offset, :], axis=1)
    out = out.astype(np.float32)

    return out


def depad_win(imdata, padding):
    if len(imdata.shape) == 3:
        _, h, w = imdata.shape
        return imdata[:, padding:h - padding, padding:w - padding]
    elif len(imdata.shape) == 2:
        h, w = imdata.shape
        return imdata[padding:h - padding, padding:w - padding]


class ImgInfo:
    def __init__(self, filename, name_only=False):
        if os.path.isfile(filename):
            self.filename = filename
            self.dataset = gdal.Open(self.filename, gdal.GA_ReadOnly)  # 打开文件
        else:
            raise FileNotFoundError("这是一个文件未找到错误: {}".format(filename))
        self.im_basename = os.path.basename(filename)
        self.im_sensor = self.parse_sensor()
        self.im_date = self.parse_date()
        if not name_only:
            self.im_width = self.dataset.RasterXSize  # 栅格矩阵的列数
            self.im_height = self.dataset.RasterYSize  # 栅格矩阵的行数
            self.im_proj = self.dataset.GetProjection()  # 地图投影信息，字符串表示
            self.im_geotrans = self.dataset.GetGeoTransform()  # 仿射矩阵，左上角像素的大地坐标和像素分辨率
            self.im_resx, self.im_resy = self.im_geotrans[1], self.im_geotrans[5]
            self.im_bands = self.dataset.RasterCount  # 波段数

    def parse_sensor(self):
        filename = self.filename.lower()

        sensors = {
            'triplesat': 'TRIPLESAT',
            'gf1b': 'GF1B',
            'gf1c': 'GF1C',
            'gf1d': 'GF1D',
            'gf1': 'GF1',
            'gf2': 'GF2',
            'gf3': 'GF3',
            'gf4': 'GF4',
            'gf5': 'GF5',
            'gf6': 'GF6',
            'gf7': 'GF7',
            'zy1': 'ZY1',
            'zy02c': 'ZY02C',
            'zy3': 'ZY3',
            'bj2': 'BJ2',
            'bj3': 'BJ3',
            'pleiades': 'PS',
            'ps': 'PS',
            'planet': 'PS',
            'pl': 'PL',
            'sv-1': 'SV1',
            'sv1': 'SV1',
            'sv-2': 'SV2',
            'sv2': 'SV2',
            'sv-3': 'SV3',
            'sv3': 'SV3',
            'sv-4': 'SV4',
            'sv4': 'SV4',
            'cb04': 'CB4',
            'cb4': 'CB4',
            'jl1': 'JL1',
            'hj': 'HJ',
            'dp': 'DP',
            'gj1': 'GJ1',
            'gj2': 'GJ2'
        }

        for key, value in sensors.items():
            if key in filename:
                return value
        return 'UN_KNOWN'

    def parse_date(self):
        date_pattern = r'(201[7-9]|202[0-7])(0[1-9]|1[0-2])(0[1-9]|[12][0-9]|3[01])'  # 匹配并验证日期
        dates = re.findall(date_pattern, os.path.basename(self.filename))
        for date in dates:
            year, month, day = map(int, date)  # 直接转换为整数
            # 使用datetime模块来验证日期的真实性
            valid_date = datetime.date(year, month, day)
            formatted_date = f"{year}-{month:02d}-{day:02d}"
            return formatted_date

    def add_para(self, input_dict: dict):
        for key, value in input_dict.items():
            setattr(self, key, value)

    def print_info(self):
        print('************** print_info ***************** print_info ********************* print_info ***************')
        for member_name, member_value in vars(self).items():
            if member_name not in ['dataset', 'im_geotrans']:
                print(f"{member_name}: {member_value}")
        print('************** print_info ***************** print_info ********************* print_info ***************')


def GenExtents(width, height, win_size, win_std=0):
    if win_std == 0:
        win_std = win_size
    frame = []
    x = 0
    y = 0
    while y < height:  # 高度方向滑窗
        if y + win_size >= height:
            y_left = height - win_size
            y_right = win_size
            y_end = True
        else:
            y_left = y
            y_right = win_size
            y_end = False

        while x < width:  # 宽度方向滑窗
            if x + win_size >= width:
                x_left = width - win_size
                x_right = win_size
                x_end = True
            else:
                x_left = x
                x_right = win_size
                x_end = False
            frame.append((x_left, y_left, x_right, y_right))
            x += win_std
            if x_end:
                break
        y += win_std
        x = 0
        if y_end:
            break
    return frame


def color_transfer_para(imdata, img_stats, ref_stats, clip=True, preserve_paper=True):
    trans = False
    if imdata.shape[0] == 3:
        trans = True
        imdata = einsum('ijk->jki', imdata)
    target = cv2.cvtColor(imdata, cv2.COLOR_RGB2LAB).astype("float32")

    # compute color statistics for the source and target images
    (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = ref_stats
    (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = img_stats

    # subtract the means from the target image
    (l, a, b) = cv2.split(target)

    l -= lMeanTar
    a -= aMeanTar
    b -= bMeanTar
    # print(l.max(), l.min(), a.max(), a.min(), b.max(), b.min())
    # print(lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc)
    # print(lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar)
    # print(lStdSrc / lStdTar, aStdSrc / aStdTar, bStdSrc / bStdTar)

    if preserve_paper:
        # scale by the standard deviations using paper proposed factor
        l *= (lStdTar / lStdSrc)
        a *= (aStdTar / aStdSrc)
        b *= (bStdTar / bStdSrc)
    else:
        # scale by the standard deviations using reciprocal of paper proposed factor
        l *= (lStdSrc / lStdTar)
        a *= (aStdSrc / aStdTar)
        b *= (bStdSrc / bStdTar)
    # add in the source mean
    l += lMeanSrc
    a += aMeanSrc
    b += bMeanSrc

    # clip/scale the pixel intensities to [0, 255] if they fall outside this range
    l = _scale_array(l, clip=clip)
    a = _scale_array(a, clip=clip)
    b = _scale_array(b, clip=clip)

    # merge the channels together and convert back to the RGB color
    # space, being sure to utilize the 8-bit unsigned integer data
    # type
    transfer = cv2.merge([l, a, b])
    transfer = cv2.cvtColor(transfer.astype('uint8'), cv2.COLOR_LAB2RGB)

    # 防止像素溢出和nodata赋值
    transfer[transfer == 255] -= 1
    transfer[imdata != 0] += 1
    transfer[imdata == 0] = 0

    if trans:
        transfer = einsum('jki->ijk', transfer)
    return transfer


def image_stats_path(image_path):
    # compute the mean and standard deviation of each channel
    # image = cv2.imread(image_path, 1)[:, :, (2, 1, 0)]
    img = IMAGE3(image_path)
    imdata = img.get_extent()
    imdata = np.moveaxis(imdata, 0, -1)
    (lMean, lStd, aMean, aStd, bMean, bStd) = image_stats(imdata)
    del img
    # return the color statistics
    return (lMean, lStd, aMean, aStd, bMean, bStd)


def image_stats(image):
    # 将图像从RGB转换为LAB颜色空间
    image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype("float32")
    # 分离L, A, B通道
    (l, a, b) = cv2.split(image)

    # 定义一个函数来安全地计算均值和标准差
    def safe_mean_std(channel):
        non_zero_channel = channel[channel != 0]
        if non_zero_channel.size == 0:  # 避免除以零
            return np.nan, np.nan  # 对于全零通道返回(0, 0)
        else:
            return np.nanmean(non_zero_channel), np.nanstd(non_zero_channel)

    # 计算每个通道的均值和标准差
    lMean, lStd = safe_mean_std(l)
    aMean, aStd = safe_mean_std(a)
    bMean, bStd = safe_mean_std(b)

    # 返回颜色统计数据
    return lMean, lStd, aMean, aStd, bMean, bStd


def _scale_array(arr, clip=True):
    """
    Trim NumPy array values to be in [0, 255] range with option of
    clipping or scaling.

    Parameters:
    -------
    arr: array to be trimmed to [0, 255] range
    clip: should array be scaled by np.clip? if False then input
        array will be min-max scaled to range
        [max([arr.min(), 0]), min([arr.max(), 255])]

    Returns:
    -------
    NumPy array that has been scaled to be in [0, 255] range
    """
    if clip:
        scaled = np.clip(arr, 0, 255)
    else:
        scale_range = (max([arr.min(), 0]), min([arr.max(), 255]))
        scaled = _min_max_scale(arr, new_range=scale_range)

    return scaled


def _min_max_scale(arr, new_range=(0, 255)):
    """
    Perform min-max scaling to a NumPy array

    Parameters:
    -------
    arr: NumPy array to be scaled to [new_min, new_max] range
    new_range: tuple of form (min, max) specifying range of
        transformed array

    Returns:
    -------
    NumPy array that has been scaled to be in
    [new_range[0], new_range[1]] range
    """
    # get array's current min and max
    mn = arr.min()
    mx = arr.max()

    # check if scaling needs to be done to be in new_range
    if mn < new_range[0] or mx > new_range[1]:
        # perform min-max scaling
        scaled = (new_range[1] - new_range[0]) * (arr - mn) / (mx - mn) + new_range[0]
    else:
        # return array if already in range
        scaled = arr

    return scaled


def minmax_stretch2range(im_data, out_range=(0, 255), in_range=None):
    if len(im_data.shape) == 3:
        if im_data.shape[0] < 10:
            for band in range(im_data.shape[0]):
                min_values = np.min(im_data[band], axis=(0, 1)) if in_range is None else in_range[0]
                max_values = np.max(im_data[band], axis=(0, 1)) if in_range is None else in_range[1]
                if max_values == 0:
                    return None
                im_data[band] = np.clip(
                    (im_data[band] - min_values) / (max_values - min_values) * out_range[1],
                    out_range[0], out_range[1])
        else:
            for band in range(im_data.shape[2]):
                min_values = np.min(im_data[..., band], axis=(0, 1)) if in_range is None else in_range[band][0]
                max_values = np.max(im_data[..., band], axis=(0, 1)) if in_range is None else in_range[band][1]
                if max_values == 0:
                    return None
                im_data[..., band] = np.clip(
                    (im_data[..., band] - min_values) / (max_values - min_values) * out_range[1],
                    out_range[0], out_range[1])
        return im_data
    if len(im_data.shape) == 2:
        min_value = np.min(im_data) if in_range is None else in_range[0]
        max_value = np.max(im_data) if in_range is None else in_range[1]
        im_data = np.clip((im_data - min_value) / (max_value - min_value) * out_range[1], out_range[0], out_range[1])
        return im_data


def percentile_stretch2range(im_data, percentile_min=2, percentile_max=98, out_range=(0, 255)):
    if np.max(im_data) < 1:
        return im_data
    min_val = np.percentile(im_data[im_data > 0], percentile_min)
    max_val = np.percentile(im_data[im_data > 0], percentile_max)
    l_data_new = (im_data - min_val) / (max_val - min_val) * out_range[1]
    l_data_new[im_data == 0] = 0
    l_data_new = np.clip(l_data_new, out_range[0], out_range[1])
    return l_data_new


def easy_blend(bg_patch, s_patch, m_patch):
    bg = einsum('ijk->jki', bg_patch)
    s = einsum('ijk->jki', s_patch)
    if len(m_patch.shape) == 3:
        m = einsum('ijk->jki', m_patch)
    else:
        m = np.stack([m_patch, m_patch, m_patch], axis=-1)

    s = s.astype(np.float32)
    m = m.astype(np.float32) / 255.
    bg = bg.astype(np.float32)

    m2 = 1. - m
    s2 = s * m
    bg2 = bg * m2
    out = bg2 + s2

    out = einsum('ijk->kij', out)

    return out


# 获取影像分辨率
def get_cellsize(imgfile):
    img_ds = gdal.Open(imgfile)
    if img_ds is None:
        raise Exception('无法打开影像{}.'.format(imgfile))
    gtf = img_ds.GetGeoTransform()
    del img_ds
    return gtf[1]


def cv_imread(file_path, read_type=-1):
    """解决无法读取中文路径的问题"""
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), read_type)
    return cv_img


def cv_show(img_path):
    img = cv_imread(img_path)
    cv2.namedWindow("cv_show")
    cv2.imshow("cv_show", img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def get_intersec_4extent_ds(ds1, ds2):
    gt1 = ds1.GetGeoTransform()
    gt2 = ds2.GetGeoTransform()

    # 四至地理坐标,
    r1 = [gt1[0], gt1[3], gt1[0] + (gt1[1] * ds1.RasterXSize), gt1[3] + (gt1[5] * ds1.RasterYSize)]
    r2 = [gt2[0], gt2[3], gt2[0] + (gt2[1] * ds2.RasterXSize), gt2[3] + (gt2[5] * ds2.RasterYSize)]

    # 公共区域的四至地理坐标
    intersection = [max(r1[0], r2[0]), min(r1[1], r2[1]), min(r1[2], r2[2]), max(r1[3], r2[3])]

    # 公共区域的四至图面坐标矩阵

    intersection_pixel_r1 = [coord_geo2ras(gt1, (intersection[0], intersection[1])),
                             coord_geo2ras(gt2, (intersection[2], intersection[3]))]
    # 相交区域宽高
    _w = int(intersection_pixel_r1[1][0]) - int(intersection_pixel_r1[0][0])
    _h = int(intersection_pixel_r1[1][1]) - int(intersection_pixel_r1[0][1])

    return [intersection[0], intersection[3], intersection[2], intersection[1]], _w, _h


def smooth_weight_map(weight_map):
    weight_map = 0.5 - 0.5 * np.cos(weight_map * np.pi)
    return weight_map


def normalize(im_data):
    min_value = np.min(im_data)
    max_value = np.max(im_data)
    if max_value == min_value:
        if max_value == 0:
            return np.zeros_like(im_data)
        else:
            return np.ones_like(im_data)
    return (im_data - min_value) / (max_value - min_value)


def crop_img_by_shp(img_path, shp_path, out_path, step_x=512, step_y=512, call_back=None, work_dir='workspace'):
    make_file(work_dir)
    reproj_shp_path = shp2img_reproj(img_path, [shp_path], work_dir)[0]
    mask_path = os.path.join(work_dir, 'tmp_mask_path_{}.tif'.format(secrets.token_hex(4)))
    dataset_shp = ogr.GetDriverByName('ESRI Shapefile').Open(reproj_shp_path, 0)
    layer = dataset_shp.GetLayer()
    img = IMAGE3(img_path)
    extent = img.get_4_extent()
    img_extent = (extent[0], extent[2], extent[1], extent[3])
    shp_extent = layer.GetExtent()
    shp_extent_new, delta = adjust_shp_extent(shp_extent, img_extent, img.im_resx)
    # print(delta)
    h_mask = img.im_height - delta[3] + delta[2]
    w_mask = img.im_width - delta[0] + delta[1]
    # print(h_mask, w_mask)
    geotrans_mask = (shp_extent_new[0], img.im_geotrans[1], img.im_geotrans[2], shp_extent_new[3],
                     img.im_geotrans[4], img.im_geotrans[5])
    img.create_img(filename=mask_path, out_bands=1, im_width=w_mask, im_height=h_mask, im_srs=img.im_srs,
                   im_geotrans=geotrans_mask, nodata=0)
    gdal.RasterizeLayer(img.output_dataset, [1], layer, burn_values=[1])
    dataset_shp.Destroy()
    # print('Rasterize done!, clipping ' + os.path.basename(img_path))

    mask = IMAGE3(img.output_dataset)
    mask.create_img(filename=out_path, out_bands=img.im_bands)
    exts = mask.gen_extents(step_x, step_y)
    length = len(exts)
    for i, ext in enumerate(exts):
        if call_back:
            call_back(i / length)
        img_ext = (ext[0] + delta[0], ext[1] + delta[3], ext[2], ext[3])
        imdata = img.get_extent(img_ext)
        mask_data = mask.get_extent(ext)
        out_data = imdata * mask_data
        mask.write_extent(out_data, ext)
    del img, mask


def get_raster_mask(img_path, shp_path, mask_path, work_dir='workspace'):
    reproj_shp_path = shp2img_reproj(img_path, [shp_path], work_dir)[0]
    dataset_shp = ogr.GetDriverByName('ESRI Shapefile').Open(reproj_shp_path, 0)
    layer = dataset_shp.GetLayer()
    img = IMAGE3(img_path)
    extent = img.get_4_extent()
    img_extent = (extent[0], extent[2], extent[1], extent[3])
    shp_extent = layer.GetExtent()
    shp_extent_new, delta = adjust_shp_extent(shp_extent, img_extent, img.im_resx)
    # print(delta)
    h_mask = img.im_height - delta[3] + delta[2]
    w_mask = img.im_width - delta[0] + delta[1]
    # print(h_mask, w_mask)
    geotrans_mask = (shp_extent_new[0], img.im_geotrans[1], img.im_geotrans[2], shp_extent_new[3],
                     img.im_geotrans[4], img.im_geotrans[5])
    img.create_img(filename=mask_path, out_bands=1, im_width=w_mask, im_height=h_mask, im_srs=img.im_srs,
                   im_geotrans=geotrans_mask, nodata=0)
    gdal.RasterizeLayer(img.output_dataset, [1], layer, burn_values=[1])
    dataset_shp.Destroy()
    del img
    print('get_raster_mask done!')


def adjust_shp_extent(shp_extent, img_extent, res):
    min_X_shp, max_X_shp, min_Y_shp, max_Y_shp = shp_extent
    min_X_img, max_X_img, min_Y_img, max_Y_img = img_extent

    ##右下为正，左上角偏移（dminx， dmaxy）,右下角偏移（dmaxx， dminy）个像素
    dminx = math.floor((min_X_shp - min_X_img) / res)
    dmaxx = math.ceil((max_X_shp - max_X_img) / res)
    dminy = - math.floor((min_Y_shp - min_Y_img) / res)
    dmaxy = - math.ceil((max_Y_shp - max_Y_img) / res)

    if dminx < 0:
        dminx = 0
    if dmaxx > 0:
        dmaxx = 0
    if dminy > 0:
        dminy = 0
    if dmaxy < 0:
        dmaxy = 0

    min_X_shp = min_X_img + dminx * res
    max_X_shp = max_X_img + dmaxx * res
    min_Y_shp = min_Y_img - dminy * res
    max_Y_shp = max_Y_img - dmaxy * res
    return (min_X_shp, max_X_shp, min_Y_shp, max_Y_shp), (dminx, dmaxx, dminy, dmaxy)


def write_extent_imdata(img: IMAGE3, extent, imdata):
    img.write_extent(extent=extent, im_data=imdata)


def gdal_resample(ref_path, src_path_list, work_dir, frmt='GTiff', res=None):
    make_file(work_dir)
    if (res is None) or (res <= 0):
        res = get_cellsize(ref_path)
    x_res = float(res)
    y_res = -x_res

    resample_output_list = []

    for i, src_path in enumerate(src_path_list):
        src_ceilisize = get_cellsize(src_path)
        if x_res == src_ceilisize:
            resample_output_list.append(src_path)
            continue
        else:
            preffix = os.path.splitext(os.path.basename(src_path))[0]
            output_path = os.path.join(work_dir, f'resample_{secrets.token_hex(4)}' + preffix + frmt2suffix(frmt))
            file_size = get_file_size_in_mb(src_path)
            if frmt != 'VRT':
                gdal.SetConfigOption('GDAL_CACHEMAX', '{}'.format(file_size))

            options = gdal.WarpOptions(
                format=frmt, dstNodata=0, srcNodata=0, xRes=x_res, yRes=y_res, multithread=True,
                creationOptions=['TILED=YES', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256'],
                warpOptions=['SKIP_NOSOURCE=YES', 'WRITE_FLUSH=YES', 'OPTIMIZE_SIZE=YES', 'GDAL_NUM_THREADS=ALL_CPUS']
            )
            ds = gdal.Warp(output_path, src_path, options=options)
            if ds is None:
                raise KeyError('resample error: {}'.format(src_path))
            resample_output_list.append(output_path)
            del ds
    return resample_output_list


def build_overview(filenameOrDs):
    if isinstance(filenameOrDs, gdal.Dataset):
        dataset = filenameOrDs
    elif os.path.isfile(filenameOrDs):
        dataset = gdal.Open(filenameOrDs)  # 打开文件
    else:
        raise KeyError('无法通过 {} 初始化'.format(filenameOrDs))
    dataset.BuildOverviews('NEAREST', [4, 8, 16, 32, 64, 128])


def img_resample(src_path, output_path, frmt='GTiff', res=None, ref_path=None, if_copy=False):
    if (res is None) or (res <= 0):
        if ref_path:
            res = get_cellsize(ref_path)
        else:
            raise KeyError('请输入正确分辨率或参考')
    x_res = float(res)
    y_res = -x_res
    src_ceilisize = get_cellsize(src_path)

    if x_res == src_ceilisize:
        if if_copy:
            img = IMAGE3(src_path)
            img.copy_image(output_path)
        else:
            return src_path
    else:
        options = gdal.WarpOptions(
            format=frmt, dstNodata=0, srcNodata=0, xRes=x_res, yRes=y_res, multithread=True,
            creationOptions=['TILED=YES', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256'],
            warpOptions=['SKIP_NOSOURCE=YES', 'WRITE_FLUSH=YES', 'OPTIMIZE_SIZE=YES', 'GDAL_NUM_THREADS=ALL_CPUS']
        )
        ds = gdal.Warp(output_path, src_path, options=options)
        if ds is None:
            raise KeyError('resample error: {}'.format(src_path))
        del ds
    return output_path

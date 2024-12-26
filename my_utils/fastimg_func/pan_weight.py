import time
from osgeo import gdal, osr
import cv2
import numpy as np
from scipy.optimize import least_squares


class Image:
    def __init__(self, img_path):
        if isinstance(img_path, str):
            self.ds = gdal.Open(img_path)
        elif isinstance(img_path, gdal.Dataset):
            self.ds = img_path
        if self.ds is not None:
            # image info
            self.path = img_path
            self.width = self.ds.RasterXSize
            self.height = self.ds.RasterYSize
            self.channel = self.ds.RasterCount
            # geo info
            self.trans = self.ds.GetGeoTransform()
            self.proj = self.ds.GetProjection()
            self.proj_srs = osr.SpatialReference()
            self.proj_srs.ImportFromWkt(self.proj)
            self.geo_srs = self.proj_srs.CloneGeogCS()
            self.wkt = self.proj_srs.ExportToWkt()
            self.epsg = self.proj_srs.GetAttrValue('AUTHORITY', 1)
            self.resx = self.trans[1]
            self.resy = self.trans[5]
            self.extent = [self.trans[0],
                           self.trans[3] + self.trans[5] * self.height,
                           self.trans[0] + self.trans[1] * self.width,
                           self.trans[3]]
            self.band_list = []
            self.band_list.extend(
                self.ds.GetRasterBand(c + 1) for c in range(self.channel))
            self.dtype = self.band_list[0].DataType
            self.output_path = None
            self.sample = []
        else:
            raise TypeError('{0} is not a image'.format(img_path))

    @staticmethod
    def insert_extent(ext1, ext2):
        min_x = max(ext1[0], ext2[0])
        min_y = max(ext1[1], ext2[1])
        max_x = min(ext1[2], ext2[2])
        max_y = min(ext1[3], ext2[3])
        if (min_x > max_x) or (min_y > max_y):
            return None
        else:
            return [min_x, min_y, max_x, max_y]

    def sample_point(self, winx, winy, size=1024, no_back=False):
        xstep = winx
        ystep = winy

        y = 0
        x = 0

        while y < self.height:
            if y + winy >= self.height:
                if no_back:
                    yoff = y
                else:
                    yoff = self.height - winy
                y_end = True
            else:
                yoff = y
                y_end = False

            while x < self.width:
                if x + winx >= self.width:
                    if no_back:
                        xoff = x
                    else:
                        xoff = self.width - winx
                    x_end = True
                else:
                    xoff = x
                    x_end = False
                x += xstep
                if yoff == 0 or xoff == 0:
                    continue
                self.sample.append([int(xoff - size / 2), int(yoff - size / 2), size, size])
                if x_end:
                    break
            y += ystep
            x = 0
            if y_end:
                break

    def align(self, src):
        align_option = gdal.WarpOptions(
            format='VRT',
            xRes=self.resx,
            yRes=self.resy,
            srcSRS=src.proj_srs,
            dstSRS=self.proj_srs,
            multithread=True,
            creationOptions=['BLOCKXSIZE=256', 'BLOCKYSIZE=256'],
            warpOptions=['SKIP_NOSOURCE=YES', 'WRITE_FLUSH=YES', 'OPTIMIZE_SIZE=YES', 'GDAL_NUM_THREADS=ALL_CPUS']
        )
        dst = gdal.Warp('', srcDSOrSrcDSTab=src.ds, options=align_option)
        return Image(dst)

    def clip(self, img):
        bounds = self.insert_extent(self.extent, img.extent)
        if bounds is not None:
            clip_option = gdal.WarpOptions(
                format='VRT', outputBounds=self.insert_extent(self.extent, img.extent),
                creationOptions=['BLOCKXSIZE=256', 'BLOCKYSIZE=256'],
                warpOptions=['SKIP_NOSOURCE=YES', 'WRITE_FLUSH=YES', 'OPTIMIZE_SIZE=YES', 'GDAL_NUM_THREADS=ALL_CPUS']
            )
            src = gdal.Warp('', self.ds, options=clip_option)
            dst = gdal.Warp('', img.ds, options=clip_option)
            return [Image(src), Image(dst)]
        else:
            raise Exception('no intersect area!')

    def imread(self, grid):
        return self.ds.ReadAsArray(grid[0], grid[1], int(grid[2]), int(grid[3]))

    def gray(self, grid):
        if self.channel > 3:
            data = self.ds.ReadAsArray(*grid, band_list=[2, 3, 4]).astype(np.float32)
            data_max = np.max(data)
            data /= data_max
            data = np.transpose(data, (1, 2, 0))
            lab = cv2.cvtColor(data, cv2.COLOR_BGR2LAB)[:, :, 0]
            lower_value = np.percentile(lab, 0.5)
            upper_value = np.percentile(lab, 99.5)
            lab -= lower_value
            lab /= (upper_value - lower_value)
            return np.clip(lab, 0, 1)
        elif self.channel == 3:
            data = self.ds.ReadAsArray(*grid, band_list=[1, 2, 3]).astype(np.float32)
            data_max = np.max(data)
            data /= data_max
            data = np.transpose(data, (1, 2, 0))
            lab = cv2.cvtColor(data, cv2.COLOR_BGR2LAB)[:, :, 0]
            lower_value = np.percentile(lab, 0.5)
            upper_value = np.percentile(lab, 99.5)
            lab -= lower_value
            lab /= (upper_value - lower_value)
            return np.clip(lab, 0, 1)
        else:
            data = self.ds.ReadAsArray(*grid, band_list=[1]).astype(np.float32)
            data_max = np.max(data)
            data /= data_max
            lower_value = np.percentile(data, 0.5)
            upper_value = np.percentile(data, 99.5)
            data -= lower_value
            data /= (upper_value - lower_value)
            return np.clip(data, 0, 1)

    def size_mb(self):
        depth = gdal.GetDataTypeSize(self.dtype)
        return (self.width * self.height * self.channel * depth) / (1024 * 1024)


def func(params, x, y):
    return np.dot(x, params) - y


def area_avg(ms_path, pan_path):
    t0 = time.time()
    ms = Image(ms_path)
    pan = Image(pan_path)
    res_ratio = np.floor(ms.resx / pan.resx)
    ms_aligen = pan.align(ms)
    pan_clip, ms_clip = pan.clip(ms_aligen)
    del ms, pan

    channel = ms_clip.channel
    win_sizex = pan_clip.width / channel
    win_sizey = pan_clip.height / channel
    base_size = int(np.sqrt(max(win_sizex, win_sizey)))
    sample_size = int(res_ratio * base_size)
    pan_clip.sample_point(win_sizex, win_sizey, sample_size)

    ms_sample = []
    pan_sample = []

    for s in pan_clip.sample:
        ms_data = ms_clip.imread(s).astype(np.float64)
        if not np.all(ms_data):
            continue
        pan_data = pan_clip.imread(s).astype(np.float64)
        pan_data_down = cv2.resize(pan_data, (base_size, base_size), interpolation=cv2.INTER_AREA)
        pan_data_up = cv2.resize(pan_data_down, (sample_size, sample_size), interpolation=cv2.INTER_NEAREST)
        weight_map = pan_data - pan_data_up
        mask = (weight_map >= -1) & (weight_map <= 1)
        pan_sample_temp = pan_data[mask]
        num_sample = len(pan_sample_temp)
        ms_sample_temp = np.zeros(shape=(num_sample, channel))

        for c in range(channel):
            ms_sample_temp[..., c] = ms_data[c][mask]

        ms_sample.append(ms_sample_temp)
        pan_sample.append(pan_sample_temp)

    ms_sample = np.concatenate(ms_sample, axis=0)
    pan_sample = np.concatenate(pan_sample, axis=0)

    bounds = (np.zeros(shape=channel), np.ones(shape=channel))
    res = least_squares(func, np.ones(shape=channel) * 0.25, args=(ms_sample, pan_sample), bounds=bounds)
    simulated_weights = np.round(res.x, 2).tolist()
    print('simulate: {0}'.format(simulated_weights))

    cov_pan = np.dot(pan_sample, pan_sample)
    put_coef = []
    for c in range(channel):
        ms_band = ms_sample[..., c]
        put_coef.append(np.dot(pan_sample, ms_band) / cov_pan)
    put_coef = np.array(put_coef)
    put_coef += res.x
    injected_weights = np.round(put_coef, 2).tolist()
    print('inject: {0}'.format(injected_weights))

    print('耗时: {0}'.format(time.time() - t0))
    return simulated_weights, injected_weights



if __name__ == '__main__':
    t0 = time.time()
    ms_path = r"D:\LCJ\tmp\ZY1E\ZY1E_VNIC_E110.4_N24.2_20221224_L1B0000546824-MUX_rpc_regis.tif"
    pan_path = r"D:\LCJ\tmp\ZY1E\ZY1E_VNIC_E110.4_N24.2_20221224_L1B0000546824-PAN_rpc.tiff"
    coef = area_avg(ms_path=ms_path, pan_path=pan_path)
    print(coef)
    print(time.time() - t0)

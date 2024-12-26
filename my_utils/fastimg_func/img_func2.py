import numpy

from fastimg_func import gdal, osr, os, np, math, ogr, proj2geo, path2frmt


class IMAGE4:
    def __init__(self, filenameOrDs, read_type=gdal.GA_ReadOnly, if_print=False):
        """
        Initialize the Image object.

        Args:
            filenameOrDs (str or gdal.Dataset): Filename or GDAL dataset.
            read_type (int): 0 for gdal.GA_ReadOnly, 1 for gdal.GA_Update.
            if_print (bool): Whether to print the image information.
        """
        self.bit_depth_lst = None
        self.output_dataset = None
        self.copy_dataset = None
        self.dataset = None
        self.bit_depth = None
        self.statis = None
        self.in_file = None
        self.copy_image_file = None

        if isinstance(filenameOrDs, gdal.Dataset):
            self.dataset = filenameOrDs
        elif os.path.isfile(filenameOrDs):
            self.in_file = filenameOrDs
            self.dataset = gdal.Open(self.in_file, read_type)  # Open the file
        else:
            raise KeyError(f'Cannot initialize IMAGE with {filenameOrDs}')

        self.read_type = read_type
        self.im_width = self.dataset.RasterXSize  # Number of columns in the raster matrix
        self.im_height = self.dataset.RasterYSize  # Number of rows in the raster matrix
        self.im_bands = self.dataset.RasterCount  # Number of bands
        self.im_geotrans = self.dataset.GetGeoTransform()  # Affine matrix, coordinates of the upper-left pixel and pixel resolution
        self.im_srs = self.dataset.GetSpatialRef()
        self.im_proj = self.dataset.GetProjection()  # Map projection information as a string
        self.im_resx, self.im_resy = self.im_geotrans[1], self.im_geotrans[5]
        self.datatype = self.dataset.GetRasterBand(1).DataType

        if if_print:
            print('*' * 200)
            print(f"in_file: {self.in_file}")
            print(f"res_x: {self.im_resx}, res_y: {self.im_resy}")
            print(f"width: {self.im_width}, height: {self.im_height}, bands: {self.im_bands}")
            print(f"im_geotrans: {self.im_geotrans}")
            print(f"im_proj: {self.im_proj}")
            print('*' * 200)

    def get_extent(self, extent=None, bands=None):
        # 默认值处理
        x, y = (0, 0) if extent is None else extent[:2]
        x_size, y_size = (self.im_width, self.im_height) if extent is None else extent[2:]

        # 边界检查和修正
        x = max(0, min(x, self.im_width - x_size))
        y = max(0, min(y, self.im_height - y_size))

        # 读取整个图像的情况
        if bands is None:
            return self.dataset.ReadAsArray(x, y, x_size, y_size)

        # 单一波段的情况
        if isinstance(bands, int):
            assert 0 <= bands < self.im_bands, f"Band index {bands} out of range"
            return self.dataset.GetRasterBand(bands + 1).ReadAsArray(x, y, x_size, y_size)

        # 多波段的情况
        if isinstance(bands, list):
            assert all(0 <= band < self.im_bands for band in bands), "One or more band indices out of range"
            out_data = [self.dataset.GetRasterBand(band + 1).ReadAsArray(x, y, x_size, y_size) for band in bands]
            return np.stack(out_data, axis=0)

    def write_extent(self, im_data=None, extent=None, target='output', bands=None):
        """
        写入影像数据到指定数据集。

        参数:
            im_data (numpy.ndarray): 要写入的数据。
            extent (tuple): 写入的范围 (x, y, x_size, y_size)。
            target (str): 目标数据集，'self' 表示写入当前数据集，'copy' 表示写入复制的数据集，'output' 表示写入输出的数据集。
            bands (int or list): 指定写入的波段（可选）。
        """
        assert im_data is not None, "im_data is None !!!"
        assert target in ['self', 'copy', 'output'], "Invalid target specified"

        # 获取目标数据集
        if target == 'output':
            dataset = self.output_dataset
            num_bands = self.out_bands
            width = self.im_width
            height = self.im_height

        elif target == 'copy':
            dataset = self.copy_dataset
            num_bands = dataset.RasterCount
            width = dataset.RasterXSize
            height = dataset.RasterYSize
        else:
            dataset = self.dataset
            num_bands = self.im_bands
            width = self.im_width
            height = self.im_height
            assert self.read_type == gdal.GA_Update, "Dataset is not opened in update mode"

        assert dataset is not None, "Target dataset is None"

        # 设置默认范围并进行边界检查和修正
        extent = extent or (0, 0, width, height)
        x, y, x_size, y_size = extent

        x = max(0, min(x, width - x_size))
        y = max(0, min(y, height - y_size))

        if num_bands == 1:
            assert im_data.ndim == 2, "Data shape does not match single band requirement"
            dataset.GetRasterBand(1).WriteArray(im_data, xoff=x, yoff=y)
        else:
            if bands is not None:
                if isinstance(bands, int):
                    assert 0 <= bands < num_bands, f"Band index {bands} out of range"
                    dataset.GetRasterBand(bands + 1).WriteArray(im_data, xoff=x, yoff=y)
                elif isinstance(bands, list):
                    assert all(0 <= band < num_bands for band in bands), "One or more band indices out of range"
                    for i, band in enumerate(bands):
                        dataset.GetRasterBand(band + 1).WriteArray(im_data[i], xoff=x, yoff=y)
            else:
                assert im_data.shape[0] == num_bands, "Data shape does not match the number of bands"
                for i in range(num_bands):
                    dataset.GetRasterBand(i + 1).WriteArray(im_data[i], xoff=x, yoff=y)

        dataset.FlushCache()

    def create_img(self, filename, out_bands=None, im_width=None, im_height=None, im_geotrans=None,
                   datatype=None, block_size=(256, 256), nodata=0, im_srs=None):
        """
        创建输出影像文件。

        参数:
            filename: 输出文件名
            out_bands: 输出影像的波段数
            im_width: 输出影像的宽度
            im_height: 输出影像的高度
            im_geotrans: 仿射变换参数
            datatype: 数据类型 (1-gdal.GDT_Byte, 2-gdal.GDT_UInt16, 6-gdal.GDT_Float32)
            block_size: 块大小 (默认256x256)
            nodata: 无效值 (默认0)
            im_srs: 空间参考系
        """
        self.output_file = filename

        # 获取驱动
        driver = gdal.GetDriverByName(self.get_frmt(filename))

        # 设置选项，除非是VRT或HFA类型
        if self.get_frmt(filename) not in ["VRT", "HFA"]:
            options = ['TILED=YES', f'BLOCKXSIZE={block_size[0]}', f'BLOCKYSIZE={block_size[1]}']
        else:
            options = []

        # 数据类型默认值
        datatype = datatype or self.datatype

        # 波段数默认值
        self.out_bands = out_bands or self.im_bands

        # 创建输出影像文件
        width = im_width or self.im_width
        height = im_height or self.im_height
        self.output_dataset = driver.Create(filename, width, height, self.out_bands, datatype, options=options)

        # 设置仿射变换参数
        geotrans = im_geotrans or self.im_geotrans
        self.output_dataset.SetGeoTransform(geotrans)

        # 设置空间参考系
        srs = im_srs or self.im_srs
        self.output_dataset.SetSpatialRef(srs)

        # 设置无效值
        if nodata is not None:
            for band in range(self.out_bands):
                self.output_dataset.GetRasterBand(band + 1).SetNoDataValue(nodata)

        self.output_dataset.FlushCache()

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

    def compute_statistics(self, if_print=False, approx_ok=True):
        """
        计算影像的统计信息，包括最小值、最大值、均值和标准差。

        参数:
            if_print (bool): 是否打印统计信息 (默认False)。
            approx_ok (bool): 是否允许近似统计 (默认True)。

        返回:
            list: 包含每个波段的统计信息的列表。
        """
        statis = []
        for i in range(self.im_bands):
            band = self.dataset.GetRasterBand(i + 1)
            stats = band.ComputeStatistics(approx_ok)
            statis.append(stats)

            if if_print:
                print(f"Band {i + 1} - min: {stats[0]}, max: {stats[1]}, mean: {stats[2]}, std: {stats[3]}", flush=True)

        self.statis = statis
        return statis

    def compute_bit_depth(self, approx_ok=True, if_print=False, multichannels=False):
        """
        计算影像的比特深度，并根据需要打印结果。

        参数:
            approx_ok (bool): 是否允许近似统计 (默认True)。
            if_print (bool): 是否打印比特深度 (默认False)。
            multichannels (bool): 是否返回多波段的比特深度列表 (默认False)。

        返回:
            int 或 list: 影像的比特深度或包含每个波段比特深度的列表。
        """
        if not hasattr(self, 'statis') or not self.statis:
            self.compute_statistics(approx_ok=approx_ok)
        bit_depth_lst = []
        for s in self.statis:
            bit_depth_lst.append(2 ** math.ceil(math.log2(s[1] + 1)))  # s[1] 是最大值
        self.bit_depth_lst = bit_depth_lst
        if multichannels:
            if if_print:
                for i, bit_depth in enumerate(bit_depth_lst):
                    print(f"Band {i + 1} bit depth: {bit_depth}", flush=True)
            return bit_depth_lst
        else:
            bit_depth = max(bit_depth_lst)
            self.bit_depth = bit_depth
            if if_print:
                print(f"Computed bit depth: {bit_depth}", flush=True)
            return bit_depth

    def __del__(self):
        self.dataset = None
        self.copy_dataset = None
        self.output_dataset = None

    def copy_image(self, filename, delete=False):
        """
        复制影像文件。如果目标文件存在且delete为False，则跳过复制。

        参数:
            filename (str): 目标文件名。
            delete (bool): 如果目标文件存在，是否删除并重新复制 (默认False)。
        """
        print(f'copy files {filename}')
        copy_frmt = path2frmt(filename)
        driver = gdal.GetDriverByName(copy_frmt)
        if os.path.exists(filename):
            if delete:
                driver.Delete(filename)
            else:
                print("delete options is False, skipping...")
                return
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        # 同类型间拷贝最快
        if self.in_file and copy_frmt == path2frmt(self.in_file):
            driver.CopyFiles(filename, self.in_file)
        else:
            gdal.Translate(filename, self.in_file, format=copy_frmt)

        self.copy_image_file = filename
        os.chmod(self.copy_image_file, 0o755)  # 设置目标文件的权限
        self.copy_dataset = gdal.Open(self.copy_image_file, gdal.GA_Update)
        # self.copy_dataset.BuildOverviews("NONE", [])
        self.copy_dataset.FlushCache()
        print(f'copy finish')

    def get_4_extent(self, dataset=None):
        """
        Calculate the extent (bounding box) of the image.

        Parameters:
            dataset (gdal.Dataset, optional): GDAL dataset to calculate extent for. If None, uses self attributes.

        Returns:
            tuple: (x_min, y_min, x_max, y_max) representing the extent of the image.
        """
        if not dataset:
            # Calculate the extent using the object's attributes
            x_min = self.im_geotrans[0]
            y_max = self.im_geotrans[3]
            x_max = x_min + self.im_geotrans[1] * self.im_width
            y_min = y_max + self.im_geotrans[5] * self.im_height
        else:
            # Get the geotransform of the provided dataset
            geotransform = dataset.GetGeoTransform()
            # Get the width and height of the provided dataset
            width = dataset.RasterXSize
            height = dataset.RasterYSize
            # Calculate the extent using the dataset's attributes
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
        """
        Determine the GDAL format based on the file extension.

        Parameters:
            img_path (str): The path to the image file.

        Returns:
            str: The GDAL format corresponding to the file extension.

        Raises:
            KeyError: If the file extension is not supported.
        """
        self.suffix = os.path.splitext(os.path.basename(img_path))[1].lower()

        if self.suffix in ['.tif', '.tiff']:
            self.frmt = 'GTiff'
        elif self.suffix in ['.img']:
            self.frmt = 'HFA'
        elif self.suffix in ['.vrt']:
            self.frmt = 'VRT'
        else:
            raise KeyError(f'{self.suffix} is not a supported file format')

        return self.frmt

    def is_projected(self):
        """
        Determine if the image's spatial reference system is projected.

        Returns:
            bool: True if the spatial reference system is projected, False otherwise.
        """
        proj_srs = osr.SpatialReference()
        proj_srs.ImportFromWkt(self.im_proj)
        return proj_srs.IsProjected()

    def pad_win(self, extent, padding):
        """
        Pad the window with specified padding.

        Parameters:
            extent (tuple): The extent to be padded (col_off, row_off, local_width, local_height).
            padding (int): The padding size.

        Returns:
            numpy.ndarray: The padded image data.
        """
        # Parse and adjust extent coordinates
        col_off, row_off, local_width, local_height = extent
        col_off -= padding
        row_off -= padding
        local_width += 2 * padding
        local_height += 2 * padding

        # Get image global attributes
        bands, global_width, global_height = self.im_bands, self.im_width, self.im_height

        # Initialize offsets
        left_offset = max(0, -col_off)
        top_offset = max(0, -row_off)
        right_offset = max(0, col_off + local_width - global_width)
        bottom_offset = max(0, row_off + local_height - global_height)

        # Calculate new boundary values
        c = max(0, col_off)
        r = max(0, row_off)
        w = min(local_width - left_offset - right_offset, global_width - c)
        h = min(local_height - top_offset - bottom_offset, global_height - r)

        # Get image data and initialize output array
        imdata = self.get_extent([c, r, w, h])
        if len(imdata.shape) == 2:
            imdata = imdata[np.newaxis, :, :]
        out = np.zeros((bands, local_height, local_width), dtype=np.float32)
        out[:, top_offset:top_offset + h, left_offset:left_offset + w] = imdata

        # Mirror padding
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

    def depad_win(self, imdata, padding):
        """
        Remove padding from the window.

        Parameters:
            imdata (numpy.ndarray): The padded image data.
            padding (int): The padding size.

        Returns:
            numpy.ndarray: The depadded image data.
        """
        _, h, w = imdata.shape
        return imdata[:, padding:h - padding, padding:w - padding]

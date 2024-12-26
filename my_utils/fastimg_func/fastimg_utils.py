import os
import secrets
import time

from osgeo import gdal, ogr
from osgeo_utils import ogr_layer_algebra
from .shp_func import split_multipolyg2poly, merge_features
from .coord_func import coord_geo2ras
from .file_func import frmt2suffix, path2frmt
from .file_func import make_file
from .img_func import ImgInfo, IMAGE3, get_intersec_4extent_ds, get_cellsize
from .reproj import shp_reproj
from .timer import program_progress, get_time_id
import secrets
gdal.SetConfigOption('SHAPE_ENCODING', 'gbk')


def generate_overview(img_path, work_dir, pct=None, size=None, name=None, frmt='GTiff'):
    print('generate_overview:{}'.format(img_path))
    if pct == 100:
        return img_path
    make_file(work_dir)
    basename = os.path.basename(img_path)
    preffix, _ = os.path.splitext(basename)
    suffix = frmt2suffix(frmt)
    if name is None:
        out_path = os.path.join(work_dir, preffix + '_overview_{}{}'.format(secrets.token_hex(4), suffix))
    else:
        out_path = os.path.join(work_dir, preffix + '_{}{}'.format(name, suffix))
    if pct is not None:
        options = gdal.TranslateOptions(format=frmt, heightPct=pct, widthPct=pct, resampleAlg=gdal.GRA_NearestNeighbour)
    elif size is not None:
        options = gdal.TranslateOptions(format=frmt, width=size[0], height=size[1], resampleAlg=gdal.GRA_NearestNeighbour)
    else:
        raise KeyError('para error')
    ds = gdal.Translate(destName=out_path, srcDS=img_path, options=options)
    if ds is None:
        raise KeyError('generate_overview error: {}'.format(img_path))
    del ds
    return out_path


def raster_mosaic(file_path_list, output_path, shp_path=None, call_back=None, work_dir=None):
    if os.path.exists(output_path):
        print(f'删除已存在文件：{output_path}')
        gdal.Unlink(output_path)
    make_file(os.path.dirname(output_path))
    assert len(file_path_list) > 1
    file_path_list = file_path_list[::-1]
    ds_list = []
    reference_file_path = file_path_list[0]
    input_file1 = gdal.Open(reference_file_path, gdal.GA_ReadOnly)
    input_proj1 = input_file1.GetProjection()
    for path in file_path_list:
        ds_list.append(gdal.Open(path, gdal.GA_ReadOnly))
    frmt = path2frmt(output_path)

    if frmt == 'HFA':
        warpOptions = ['SKIP_NOSOURCE=YES', 'WRITE_FLUSH=YES', 'GDAL_NUM_THREADS=ALL_CPUS']
        creationOptions = ['TILED=YES']
    else:
        warpOptions = ['SKIP_NOSOURCE=YES', 'WRITE_FLUSH=YES', 'OPTIMIZE_SIZE=YES', 'GDAL_NUM_THREADS=ALL_CPUS']
        creationOptions = ['TILED=YES', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256']
    if shp_path:
        work_dir = os.path.join(os.path.dirname(output_path),
                                'ras_mos_{}'.format(secrets.token_hex(4))) if not work_dir else work_dir
        make_file(work_dir)
        new_shp_path = os.path.join(work_dir, os.path.basename(shp_path))
        merge_features(shp_path, new_shp_path)
        options = gdal.WarpOptions(dstSRS=input_proj1, format=frmt, srcNodata=0, dstNodata=0,
                                   creationOptions=creationOptions, warpOptions=warpOptions,
                                   multithread=True, callback=call_back, cutlineDSName=new_shp_path, cropToCutline=True)
    else:
        options = gdal.WarpOptions(dstSRS=input_proj1, format=frmt, srcNodata=0, dstNodata=0,
                                   creationOptions=creationOptions, warpOptions=warpOptions,
                                   multithread=True, callback=call_back)
    out_ds = gdal.Warp(output_path, ds_list, options=options)

    if not out_ds:
        raise KeyError('raster_mosaic error: {}'.format(file_path_list))
    else:
        del input_file1, out_ds
        del ds_list


class ImgInfoSub(ImgInfo):
    def __init__(self, filename, cloud_shp=None, valid_shp=None, rem_path=None):
        super().__init__(filename)
        self.cloud_shp = cloud_shp
        self.valid_shp = valid_shp
        self.rem_path = rem_path


def shp2raster(shp_path, ref_ras_path, target_ras_path, attribute_field=''):
    print("shp2raster:{}".format(shp_path))
    make_file(os.path.dirname(target_ras_path))
    ref_tif_file = IMAGE3(ref_ras_path)
    ref_tif_file.create_img(
        filename=target_ras_path,
        im_width=ref_tif_file.im_width, im_height=ref_tif_file.im_height,
        im_proj=ref_tif_file.im_proj, im_geotrans=ref_tif_file.im_geotrans,
        out_bands=1,
        datatype=gdal.GDT_Byte
    )

    shp_file = ogr.Open(shp_path)
    shp_layer = shp_file.GetLayer()
    gdal.RasterizeLayer(
        dataset=ref_tif_file.output_dataset,
        bands=[1],
        layer=shp_layer,
        # options=[f"ATTRIBUTE={attribute_field}"]
    )
    del ref_tif_file.output_dataset
    return target_ras_path


def clip_shp_by_region(to_clip_shp, out_region_shp, output_shp, work_dir):
    reproj = shp_reproj(to_clip_shp, [out_region_shp], work_dir)[0]
    opt = [
        ' ',
        '-input_ds', "{}".format(to_clip_shp),
        '-method_ds', "{}".format(reproj),
        '-output_ds', "{}".format(output_shp),
        "-overwrite", 'Clip'
    ]
    ogr_layer_algebra.main(opt)


class UnionShpSub:
    def __init__(self, filename):
        assert os.path.isfile(filename)
        assert os.path.splitext(filename)[1] == '.shp'
        self.filename = filename
        self.datasets = ogr.Open(filename, 1)
        if self.datasets is None:
            raise Exception("无法打开Shapefile数据集。")
        self.img_info_lst = None
        self.img_lst = []
        self.valid_lst = []
        self.cloud_lst = []
        self.rem_lst = []

    def parse_union_shp(self):
        self.img_info_lst = []
        for layer in self.datasets:
            for feat in layer:
                img_path = feat.GetField('FileName')
                try:
                    rem_path = feat.GetField('RemPath')
                except:
                    rem_path = img_path
                img_info = ImgInfoSub(img_path, cloud_shp=feat.GetField('CloudPath'),
                                      valid_shp=feat.GetField('ValidPath'), rem_path=rem_path)
                self.img_info_lst.append(img_info)

    def set_cloud_removal_path(self, output_dir=None):
        assert output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for layer in self.datasets:
            if layer.FindFieldIndex('RemPath', True) < 0:
                print('创建RemPath字段...')
                field_defn = ogr.FieldDefn('RemPath', ogr.OFTString)
                field_defn.SetWidth(254)
                layer.CreateField(field_defn)
            for feat in layer:
                rem_path = os.path.join(output_dir, 'rem_' + os.path.basename(feat.GetField('FileName')))
                feat.SetField('RemPath', rem_path)
                layer.SetFeature(feat)

    def get_img_and_shp_lst(self, if_print=False):
        self.parse_union_shp()
        assert len(self.img_info_lst) > 0
        for imginfo in self.img_info_lst:
            if if_print:
                print('***********************************************************')
                print('img_path:{}'.format(imginfo.filename))
                print('valid_shp:{}'.format(imginfo.valid_shp))
                print('cloud_shp:{}'.format(imginfo.cloud_shp))
                print('rem_path:{}'.format(imginfo.rem_path))
                print('***********************************************************')
            self.img_lst.append(imginfo.filename)
            self.valid_lst.append(imginfo.valid_shp)
            self.cloud_lst.append(imginfo.cloud_shp)
            self.rem_lst.append(imginfo.rem_path)
        return self.img_lst, self.valid_lst, self.cloud_lst, self.rem_lst

    def __del__(self):
        self.datasets = None


def sort_shp_img_lst(shp_lst, sort_type):
    # todo 指定分辨率下的时相优先，指定时相下的分辨率优先， 待做
    # 仅支持只有一个feature的shp, 仅支持det_cloud生产的shp， valid和cloud均可
    img_info_lst = []
    for filename in shp_lst:
        assert os.path.isfile(filename)
        assert os.path.splitext(filename)[1] == '.shp'
        datasets = ogr.Open(filename)
        if datasets is None:
            raise Exception("无法打开Shapefile数据集。")
        for layer in datasets:
            for i, feat in enumerate(layer):
                img_info = ImgInfoSub(filename=feat.GetField('FileName'),
                                      cloud_shp=feat.GetField('CloudPath'),
                                      valid_shp=feat.GetField('ValidPath'))
                img_info_lst.append(img_info)
                break
    if sort_type == 'date':
        sorted_imginfo_list = sorted(img_info_lst, key=lambda x: x.im_date, reverse=True)
    elif sort_type == 'res':
        sorted_imginfo_list = sorted(img_info_lst, key=lambda x: x.im_resx)
    else:
        print('不进行排序')
        sorted_imginfo_list = img_info_lst

    sorted_valid_lst = []
    sorted_cloud_lst = []
    sorted_img_lst = []
    for img_info in sorted_imginfo_list:
        sorted_valid_lst.append(img_info.valid_shp)
        sorted_cloud_lst.append(img_info.cloud_shp)
        sorted_img_lst.append(img_info.filename)
    return sorted_img_lst, sorted_valid_lst, sorted_cloud_lst


# 擦除
def erase_shp_by_mask(to_erase, erase_list, erase_out_dir, inter_out_dir, image_path_list=None, work_dir=None):
    make_file(erase_out_dir)
    make_file(inter_out_dir)
    # 开始批量擦除
    erase_path_list = []
    inter_path_list = []
    new_img_path_list = []
    driver = ogr.GetDriverByName('ESRI Shapefile')
    to_erase_split = os.path.join(work_dir, os.path.basename(to_erase).replace('.shp', '_split.shp'))
    split_multipolyg2poly(to_erase, to_erase_split)

    for i, erase_path in enumerate(erase_list):
        basename = os.path.basename(erase_path)
        # 被擦除shp
        to_erase_shp = driver.Open(to_erase)
        to_erase_layer = to_erase_shp.GetLayer()
        num_feature = to_erase_layer.GetFeatureCount()
        if num_feature == 0:
            break

        # 被擦除shp_split
        to_erase_split_shp = driver.Open(to_erase_split)
        to_erase_split_layer = to_erase_split_shp.GetLayer()

        dst_erase = os.path.join(erase_out_dir, f'erase{secrets.token_hex(4)}_' + basename)
        erase_path_list.append(dst_erase)
        dst_inter = os.path.join(inter_out_dir, f'inter{secrets.token_hex(4)}_' + basename)
        inter_path_list.append(dst_inter)

        test_shp = driver.Open(erase_path, 0)
        test_layer = test_shp.GetLayer()
        test_srs = test_layer.GetSpatialRef()
        test_defn = test_layer.GetLayerDefn()

        outds_inter = driver.CreateDataSource(dst_inter)
        time.sleep(0.1)
        outlayer_inter = outds_inter.CreateLayer(dst_inter, srs=test_srs, geom_type=ogr.wkbPolygon)

        outds_erase = driver.CreateDataSource(dst_erase)
        time.sleep(0.1)
        outlayer_erase = outds_erase.CreateLayer(dst_erase, srs=test_srs, geom_type=ogr.wkbPolygon)

        for j in range(test_defn.GetFieldCount()):
            outlayer_inter.CreateField(test_defn.GetFieldDefn(j))
            outlayer_erase.CreateField(test_defn.GetFieldDefn(j))

        # 获取擦除剩余和擦除部分
        if i == 0:
            try:
                to_erase_layer.Erase(test_layer, outlayer_erase)
                to_erase_split_layer.Intersection(test_layer, outlayer_inter)
                to_erase_shp.Destroy()
                to_erase_split_shp.Destroy()
            except:
                pass
        else:
            try:
                tmp_shp = driver.Open(erase_path_list[i - 1], 1)
                tmp_layer = tmp_shp.GetLayer()
                tmp_feat_count = tmp_layer.GetFeatureCount()
                if tmp_feat_count == 0:
                    break
                tmp_split_shp_path = os.path.join(work_dir,
                                                  os.path.basename(erase_path_list[i - 1]).replace('.shp', '_split.shp'))
                split_multipolyg2poly(erase_path_list[i - 1], tmp_split_shp_path)
                tmp_split_shp = driver.Open(tmp_split_shp_path, 1)
                tmp_split_layer = tmp_split_shp.GetLayer()

                tmp_layer.Erase(test_layer, outlayer_erase)
                tmp_split_layer.Intersection(test_layer, outlayer_inter)
                tmp_shp.Destroy()
                tmp_split_shp.Destroy()
            except:
                pass
        if image_path_list:
            # 不相交的不输出
            if outlayer_inter.GetFeatureCount() != 0:
                new_img_path_list.append(image_path_list[i])
            else:
                inter_path_list.pop()

        # 擦除完毕
        if outlayer_erase.GetFeatureCount() == 0:
            break
    if image_path_list:
        return inter_path_list, new_img_path_list
    else:
        return inter_path_list


# # 擦除
def erase_shp_by_mask_for_union(to_erase_path, erase_list, erase_out_dir, inter_out_dir):
    make_file(erase_out_dir)
    make_file(inter_out_dir)
    # 开始批量擦除
    remain_path_list = []
    inter_path_list = []
    driver = ogr.GetDriverByName('ESRI Shapefile')
    for i, erase_path in enumerate(erase_list):
        dst_erase = os.path.join(erase_out_dir, f'erase_{i}.shp')
        remain_path_list.append(dst_erase)
        dst_inter = os.path.join(inter_out_dir, f'inter_{i}.shp')
        inter_path_list.append(dst_inter)

        erase_shp = driver.Open(erase_path, 0)
        erase_layer = erase_shp.GetLayer()
        erase_srs = erase_layer.GetSpatialRef()
        erase_defn = erase_layer.GetLayerDefn()

        outds_inter = driver.CreateDataSource(dst_inter)
        outlayer_inter = outds_inter.CreateLayer(dst_inter, srs=erase_srs, geom_type=ogr.wkbPolygon)
        outds_erase = driver.CreateDataSource(dst_erase)
        outlayer_erase = outds_erase.CreateLayer(dst_erase, srs=erase_srs, geom_type=ogr.wkbPolygon)
        defns = []
        for j in range(erase_defn.GetFieldCount()):
            defn = erase_defn.GetFieldDefn(j)
            outlayer_inter.CreateField(defn)
            outlayer_erase.CreateField(defn)
            defns.append(defn)
        # 被擦除shp
        now_to_erase_path = to_erase_path if i == 0 else remain_path_list[i - 1]
        # 获取擦除剩余和擦除部分
        to_erase_shp = driver.Open(now_to_erase_path, 0)
        try:
            to_erase_layer = to_erase_shp.GetLayer()
        except:
            to_erase_layer = to_erase_shp.GetLayer()
        to_erase_feat_count = to_erase_layer.GetFeatureCount()
        if to_erase_feat_count == 0:
            print("擦除完毕")
            break

        a = to_erase_layer.Erase(erase_layer, outlayer_erase)
        b = to_erase_layer.Intersection(erase_layer, outlayer_inter)

        outds_inter.Destroy()
        outds_erase.Destroy()
        to_erase_shp.Destroy()

        if a + b > 0:
            remain_path_list[i] = remain_path_list[i - 1]
            os.remove(dst_inter)
            os.remove(dst_erase)
            inter_path_list.remove(dst_inter)

    return inter_path_list


def find_extent_ref2src(ref_path, src_path, outpath, work_dir, resample_method=gdal.GRA_Bilinear,
                        res2src=True):
    print('get inter extent img ...')
    make_file(work_dir)
    reference_ds = gdal.Open(ref_path, gdal.GA_ReadOnly)
    source_ds = gdal.Open(src_path, gdal.GA_ReadOnly)
    source_trans = source_ds.GetGeoTransform()
    srcSrs = source_ds.GetProjectionRef()
    if res2src:
        dstResX, dstResY = source_trans[1], source_trans[5]
    else:
        ref_trans = reference_ds.GetGeoTransform()
        dstResX, dstResY = ref_trans[1], ref_trans[5]
    tmp_path = os.path.join(work_dir, '1warp{}.vrt'.format(secrets.token_hex(4)))
    ref2src = gdal.WarpOptions(
        format=path2frmt(tmp_path), xRes=dstResX, yRes=dstResY,
        targetAlignedPixels=True, resampleAlg=resample_method, dstSRS=srcSrs,
        warpOptions=['SKIP_NOSOURCE=YES', 'OPTIMIZE_SIZE=YES', 'GDAL_NUM_THREADS=ALL_CPUS']
    )
    ref_warp = gdal.Warp(destNameOrDestDS=tmp_path, srcDSOrSrcDSTab=reference_ds, options=ref2src)

    if not ref_warp:
        raise KeyError('get inter extent first warp error')
    else:
        bounds, _w, _h = get_intersec_4extent_ds(ref_warp, source_ds)
        # if _w < 0 or _h <= 0:
        #     raise AssertionError(f"no intersect: \n{src_path}, {ref_path}")
        del ref_warp, ref2src

    creationOptions = ['TILED=YES', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256'] if path2frmt(outpath) != 'VRT' else []
    ref2inter = gdal.WarpOptions(
        outputBounds=bounds, resampleAlg=resample_method, errorThreshold=dstResX,
        creationOptions=creationOptions,
        warpOptions=['SKIP_NOSOURCE=YES', 'WRITE_FLUSH=YES', 'OPTIMIZE_SIZE=YES', 'GDAL_NUM_THREADS=ALL_CPUS'])
    inter_warp = gdal.Warp(destNameOrDestDS=outpath, srcDSOrSrcDSTab=tmp_path, options=ref2inter)

    if inter_warp is None:
        raise KeyError('get inter extent second warp error')
    else:
        # 输出相交区域的影像路径 和 左上角栅格坐标
        geo_trans = inter_warp.GetGeoTransform()
        x_geo, y_geo = geo_trans[0], geo_trans[3]
        lu_pt = coord_geo2ras(reference_ds.GetGeoTransform(), [x_geo, y_geo])
        print('get inter extent img finish!')
        del inter_warp, reference_ds, source_ds, ref2inter
        return outpath, lu_pt


def find_extent_4match(ref_path, src_path, ref_warp, src_warp, resample_method=gdal.GRA_Bilinear):
    # print('get inter extent img ...')
    src_ds = gdal.Open(src_path, gdal.GA_ReadOnly)
    src_trans = src_ds.GetGeoTransform()
    ref_ds = gdal.Open(ref_path, gdal.GA_ReadOnly)
    ref_trans = ref_ds.GetGeoTransform()
    ref_resx, ref_resy = ref_trans[1], ref_trans[5]
    dstSrs = ref_ds.GetSpatialRef()
    src_resx, src_resy = src_trans[1], src_trans[5]
    dstResX, dstResY = max(src_resx, ref_resx), min(src_resy, ref_resy)

    if dstResX == src_resx:
        res_type = 'src'
    else:
        res_type = 'ref'

    opt1 = gdal.WarpOptions(
        xRes=dstResX, yRes=dstResY, errorThreshold=0.125,
        targetAlignedPixels=True,
        resampleAlg=resample_method, dstSRS=dstSrs
    )
    warp1_ds = gdal.Warp(destNameOrDestDS=src_warp, srcDSOrSrcDSTab=src_ds, options=opt1)

    if not warp1_ds:
        raise KeyError('get inter extent first warp error')
    else:
        bounds, _, _ = get_intersec_4extent_ds(warp1_ds, ref_ds)

    opt2 = gdal.WarpOptions(
        outputBounds=bounds, resampleAlg=resample_method,
        xRes=dstResX, yRes=dstResY, errorThreshold=0.125)
    warp2_ds = gdal.Warp(destNameOrDestDS=ref_warp, srcDSOrSrcDSTab=ref_ds, options=opt2)

    # print('get inter extent img finish !')
    del warp1_ds, warp2_ds
    return res_type


# def find_extent_src2ref(reference_path, source_path, work_dir):
#     target_ds = gdal.Open(reference_path, gdal.GA_ReadOnly)
#     source_ds = gdal.Open(source_path, gdal.GA_ReadOnly)
#
#     target_trans = target_ds.GetGeoTransform()
#     dstResX = abs(target_trans[1])
#     dstResY = abs(target_trans[5])
#     dstSrs = target_ds.GetSpatialRef()
#
#     source2target = {
#         'destNameOrDestDS': '',
#         'srcDSOrSrcDSTab': source_ds,
#         'format': 'VRT',
#         'xRes': dstResX,
#         'yRes': dstResY,
#         'targetAlignedPixels': True,
#         'dstSRS': dstSrs,
#         'resampleAlg': 'Bilinear',
#         'errorThreshold': dstResX
#     }
#     source_temp = gdal.Warp(**source2target)
#     bounds, dstWidth, dstHeight = get_intersec_4extent_ds(target_ds, source_temp)
#     print("相交区域：{}X{}".format(dstWidth, dstHeight))
#     del source_temp
#
#     t = int(time.time())
#     reference_warp_path = os.path.join(work_dir, 'reference{}.tif'.format(t))
#     source_warp_path = os.path.join(work_dir, 'source{}.tif'.format(t))
#     target_opt = {
#         'destNameOrDestDS': reference_warp_path,
#         'srcDSOrSrcDSTab': target_ds,
#         'format': 'VRT',
#         'outputBounds': bounds,
#         'resampleAlg': gdal.GRA_Bilinear,
#         'width': dstWidth,
#         'height': dstHeight,
#         'errorThreshold': dstResX
#     }
#     source_opt = {
#         'destNameOrDestDS': source_warp_path,
#         'srcDSOrSrcDSTab': source_ds,
#         'format': 'VRT',
#         'xRes': dstResX,
#         'yRes': dstResY,
#         'targetAlignedPixels': True,
#         'dstSRS': dstSrs,
#         'resampleAlg': gdal.GRA_Bilinear,
#         'outputBoundsSRS': dstSrs,
#         'outputBounds': bounds,
#         'errorThreshold': dstResX
#     }
#     target_warp = gdal.Warp(**target_opt)
#     source_warp = gdal.Warp(**source_opt)
#
#     if not target_warp or not source_warp:
#         raise KeyError('crop error')
#     return reference_warp_path, source_warp_path


def batch_erase_union(shp_path_lst, out_shp_path, work_dir, out_region=None):
    # 按顺序优先级
    assert len(shp_path_lst) > 1
    make_file(os.path.dirname(out_shp_path))
    dst_shp_path = None
    wdir = os.path.join(work_dir, 'batch_erase{}'.format(secrets.token_hex(4)))
    make_file(wdir)
    for i, shp_path in enumerate(shp_path_lst[::-1]):
        if i == 0:
            dst_shp_path = shp_path
            continue
        elif i == len(shp_path_lst) - 1:
            if out_region:
                tmp_out_shp_path = os.path.join(wdir, '{}.shp'.format(i))
                run_one_erase_union(
                    dst_shp_path=dst_shp_path, src_shp_path=shp_path, out_shp_path=tmp_out_shp_path,
                    work_dir=wdir)
                clip_shp_by_region(tmp_out_shp_path, out_region, out_shp_path, work_dir)
            else:
                run_one_erase_union(
                    dst_shp_path=dst_shp_path, src_shp_path=shp_path, out_shp_path=out_shp_path,
                    work_dir=wdir)

        else:
            tmp_out_shp_path = os.path.join(wdir, '{}.shp'.format(i))
            run_one_erase_union(
                dst_shp_path=dst_shp_path, src_shp_path=shp_path, out_shp_path=tmp_out_shp_path,
                work_dir=os.path.join(wdir, 'one_erase{}'.format(i)))
            #  0424内网测试
            if os.path.exists(tmp_out_shp_path):
                dst_shp_path = tmp_out_shp_path
            #  0424内网测试


def run_one_erase_union(dst_shp_path, src_shp_path, out_shp_path, work_dir):
    make_file(os.path.dirname(out_shp_path))
    driver = ogr.GetDriverByName('ESRI Shapefile')
    # 被擦除shp
    dst_shp = driver.Open(dst_shp_path)
    dst_layer = dst_shp.GetLayer()
    num_feature = dst_layer.GetFeatureCount()
    make_file(work_dir)
    # 合并擦子
    new_src_shp_path = os.path.join(work_dir, 'cazi{}_{}'.format(secrets.token_hex(4), os.path.basename(src_shp_path)))
    merge_features(src_shp_path, new_src_shp_path)

    src_shp = driver.Open(new_src_shp_path)
    src_layer = src_shp.GetLayer()

    if num_feature == 0:
        print('num_feature = 0')
        return
    dst_srs = dst_layer.GetSpatialRef()
    dst_defn = dst_layer.GetLayerDefn()

    out_shp = driver.CreateDataSource(out_shp_path)
    time.sleep(0.1)
    out_layer = out_shp.CreateLayer(out_shp_path, srs=dst_srs, geom_type=ogr.wkbPolygon)

    make_file(work_dir)
    tmp_erase_path = os.path.join(work_dir, 'tmpshp_{}.shp'.format(secrets.token_hex(4)))

    outds_erase = driver.CreateDataSource(tmp_erase_path)
    time.sleep(0.1)
    outlayer_erase = outds_erase.CreateLayer(tmp_erase_path, srs=dst_srs, geom_type=ogr.wkbPolygon)

    FieldDefns = []
    for j in range(dst_defn.GetFieldCount()):
        out_layer.CreateField(dst_defn.GetFieldDefn(j))
        outlayer_erase.CreateField(dst_defn.GetFieldDefn(j))
        FieldDefns.append(dst_defn.GetFieldDefn(j))

    dst_layer.Erase(src_layer, outlayer_erase)

    # union src
    src_shp = driver.Open(src_shp_path)
    src_layer = src_shp.GetLayer()
    for feature in src_layer:
        geom = feature.GetGeometryRef()
        out_feat = ogr.Feature(out_layer.GetLayerDefn())
        out_feat.SetGeometry(geom)
        for FieldDefn in FieldDefns:
            out_feat.SetField(FieldDefn.GetName(), feature.GetField(FieldDefn.GetName()))
        out_layer.CreateFeature(out_feat)

    # union 擦除剩余
    outlayer_erase.ResetReading()
    for feature in outlayer_erase:
        geom = feature.GetGeometryRef()
        out_feat = ogr.Feature(out_layer.GetLayerDefn())
        out_feat.SetGeometry(geom)
        for FieldDefn in FieldDefns:
            out_feat.SetField(FieldDefn.GetName(), feature.GetField(FieldDefn.GetName()))
        out_layer.CreateFeature(out_feat)

    dst_shp.Destroy()
    out_shp.Destroy()
    src_shp.Destroy()
    outds_erase.Destroy()

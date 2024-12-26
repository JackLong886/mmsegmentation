__all__ = ['get_shp_proj', 'reproj_img2ref', 'reproj_img2ref_batch', 'reproj_img_by_shp', 'reproj_img_by_shp_batch',
           'shp2img_reproj', 'shp_reproj']

import os
import os.path
import secrets
import time

from osgeo import ogr, osr, gdal

from . import make_file, path2frmt, frmt2suffix


def get_shp_proj(shapefile_path, if_print=False):
    # 打开Shapefile文件
    driver = ogr.GetDriverByName('ESRI Shapefile')
    dataSource = driver.Open(shapefile_path, 0)  # 0 表示只读模式

    if dataSource is None:
        print('无法打开Shapefile文件')
    else:
        # 获取第一个图层
        layer = dataSource.GetLayer()

        # 获取图层的投影
        spatial_ref = layer.GetSpatialRef()
        if spatial_ref is not None:
            if if_print:
                # 打印投影信息
                print("投影信息:")
                print(spatial_ref.ExportToPrettyWkt())  # 以可读方式打印投影信息
            return spatial_ref.ExportToWkt()
        else:
            raise ValueError('该图层没有定义投影信息')
    # 关闭数据源
    dataSource = None


def reproj_img2ref(ref_path, src_path, out_path):
    print('start reproj: {}'.format(src_path))
    make_file(os.path.dirname(out_path))
    ref_ds = gdal.Open(ref_path, gdal.GA_ReadOnly)
    dstSrs = ref_ds.GetProjectionRef()

    src_ds = gdal.Open(src_path, gdal.GA_ReadOnly)
    srcSrs = src_ds.GetProjectionRef()
    if dstSrs != srcSrs:
        frmt = path2frmt(out_path)
        create_options = ['TILED=YES'] if frmt == 'HFA' else ['TILED=YES', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256']
        src2ref = gdal.WarpOptions(
            format=frmt, dstSRS=dstSrs, errorThreshold=0.125,
            creationOptions=create_options, resampleAlg=gdal.GRA_Bilinear,
            warpOptions=['SKIP_NOSOURCE=YES', 'OPTIMIZE_SIZE=YES', 'GDAL_NUM_THREADS=ALL_CPUS'])
        src_warp = gdal.Warp(destNameOrDestDS=out_path, srcDSOrSrcDSTab=src_path, options=src2ref)
        if not src_warp:
            raise KeyError('reproj error: {}'.format(src_path))
        print('finish reproj')
        del src_warp, ref_ds, src_ds
        return out_path
    else:
        del ref_ds, src_ds
        return src_path


def reproj_img2ref_batch(ref_path, src_path_list, out_path_list=None, work_dir=None, frmt=None):
    make_file(work_dir)
    if out_path_list is None:
        if work_dir is None:
            raise KeyError('cant find output_path')
        else:
            out_path_list = []
            for src_path in src_path_list:
                out_path_list.append(os.path.join(work_dir, f'reproj_{secrets.token_hex(4)}' + frmt2suffix(frmt)))
    else:
        assert len(out_path_list) == len(src_path_list)
    reproj_img_list = []
    for src_path, out_path in zip(src_path_list, out_path_list):
        reproj_img_list.append(reproj_img2ref(ref_path, src_path, out_path))
    return reproj_img_list


def reproj_img_by_shp(ref_shp_path, src_path, out_path):
    make_file(os.path.dirname(out_path))
    print('start reproj: {}'.format(src_path))
    dstSrs = get_shp_proj(shapefile_path=ref_shp_path)
    src_ds = gdal.Open(src_path, gdal.GA_ReadOnly)
    srcSrs = src_ds.GetProjectionRef()
    if dstSrs != srcSrs:
        create_options = ['TILED=YES'] if path2frmt(out_path) == 'HFA' \
            else ['TILED=YES', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256']
        src2ref = gdal.WarpOptions(
            dstSRS=dstSrs, srcSRS=srcSrs,
            creationOptions=create_options,
            warpOptions=['SKIP_NOSOURCE=YES', 'OPTIMIZE_SIZE=YES', 'GDAL_NUM_THREADS=ALL_CPUS']
        )
        src_warp = gdal.Warp(destNameOrDestDS=out_path, srcDSOrSrcDSTab=src_path, options=src2ref)
        if not src_warp:
            raise KeyError('reproj error: {}'.format(src_path))
        del src_warp, src_ds
        return out_path
    else:
        del src_ds
        return src_path


def reproj_img_by_shp_batch(ref_shp_path, src_path_list, out_path_list=None, work_dir=None, frmt='VRT'):
    make_file(work_dir)
    if out_path_list is None:
        if work_dir is None:
            raise KeyError('cant find output_path')
        else:
            out_path_list = []
            for src_path in src_path_list:
                out_path_list.append(os.path.join(work_dir, 'reproj_' +
                                                  os.path.splitext(os.path.basename(src_path))[0] + frmt2suffix(frmt)))
    else:
        assert len(out_path_list) == len(src_path_list)
    reproj_img_list = []
    for src_path, out_path in zip(src_path_list, out_path_list):
        reproj_img_list.append(reproj_img_by_shp(ref_shp_path, src_path, out_path))
    return reproj_img_list


def shp_reproj(ref_shp_path, shp_path_list, out_dir):
    '''
    字段名和值参考shp_path_list，投影参考ref_shp_path
    Args:
        ref_shp_path:
        shp_path_list:
        out_dir:

    Returns:

    '''
    driver = ogr.GetDriverByName('ESRI Shapefile')
    make_file(out_dir)
    ref_shp = driver.Open(ref_shp_path)
    ref_layer = ref_shp.GetLayer()
    num_feat = ref_layer.GetFeatureCount()
    if num_feat == 0:
        print('num_feat = 0')
        return
    ref_srs = ref_layer.GetSpatialRef()
    # ref_defn = ref_layer.GetLayerDefn()
    output_lst = []
    for shp_path in shp_path_list:
        out_path = os.path.join(out_dir, f'proj{secrets.token_hex(4)}.shp')
        src_ds = driver.Open(shp_path)
        src_layer = src_ds.GetLayer()
        num_feat = src_layer.GetFeatureCount()
        src_srs = src_layer.GetSpatialRef()
        if num_feat == 0 or src_srs == ref_srs:
            output_lst.append(shp_path)
        else:
            output_lst.append(out_path)
        outds = driver.CreateDataSource(out_path)
        time.sleep(0.1)
        outlayer = outds.CreateLayer('fastimg_layer', srs=ref_srs, geom_type=ogr.wkbPolygon)
        input_layer_defn = src_layer.GetLayerDefn()
        for j in range(input_layer_defn.GetFieldCount()):
            outlayer.CreateField(input_layer_defn.GetFieldDefn(j))
        transform = osr.CoordinateTransformation(src_srs, ref_srs)
        src_layer.ResetReading()
        for feature in src_layer:
            geom = feature.GetGeometryRef()
            geom.Transform(transform)
            out_feat = ogr.Feature(outlayer.GetLayerDefn())
            for i in range(input_layer_defn.GetFieldCount()):
                field_name = input_layer_defn.GetFieldDefn(i).GetName()
                field_value = feature.GetField(field_name)
                # 将字段名和值设置到新特征中
                out_feat.SetField(field_name, field_value)
            out_feat.SetGeometry(geom)
            outlayer.CreateFeature(out_feat)

    return output_lst


def shp2img_reproj(ref_img_path, shp_path_list, out_dir):
    driver = ogr.GetDriverByName('ESRI Shapefile')
    make_file(out_dir)
    ref_srs = gdal.Open(ref_img_path).GetSpatialRef()
    output_lst = []
    for shp_path in shp_path_list:
        out_path = os.path.join(out_dir, f'{secrets.token_hex(4)}.shp')
        output_lst.append(out_path)
        outds = driver.CreateDataSource(os.path.abspath(out_path))
        time.sleep(0.1)
        outlayer = outds.CreateLayer(out_dir, srs=ref_srs, geom_type=ogr.wkbPolygon)
        src_ds = driver.Open(shp_path)
        src_layer = src_ds.GetLayer()

        input_layer_defn = src_layer.GetLayerDefn()
        for j in range(input_layer_defn.GetFieldCount()):
            outlayer.CreateField(input_layer_defn.GetFieldDefn(j))

        src_srs = src_layer.GetSpatialRef()
        transform = osr.CoordinateTransformation(src_srs, ref_srs)
        src_layer.ResetReading()
        for feature in src_layer:
            geom = feature.GetGeometryRef()
            geom.Transform(transform)
            out_feat = ogr.Feature(outlayer.GetLayerDefn())

            for i in range(input_layer_defn.GetFieldCount()):
                field_name = input_layer_defn.GetFieldDefn(i).GetName()
                field_value = feature.GetField(field_name)
                # 将字段名和值设置到新特征中
                out_feat.SetField(field_name, field_value)

            out_feat.SetGeometry(geom)
            outlayer.CreateFeature(out_feat)

        # 关闭源数据源
        src_ds = None
        outds = None
    return output_lst

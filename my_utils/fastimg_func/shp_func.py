import os
import shutil
import time

from osgeo import ogr, gdal

from . import make_file

gdal.SetConfigOption('SHAPE_ENCODING', 'gbk')


def if_4ext_inrersect(extent1, extent2):
    if extent1[0] < extent2[1] and extent1[1] > extent2[0] and extent1[2] < extent2[3] and extent1[3] > extent2[2]:
        return True
    else:
        return False


def if_shp4ext_intersect(shp_path1, shp_path2):
    # 打开第一个矢量数据集
    ds1 = ogr.Open(shp_path1)
    if ds1 is None:
        print(f"无法打开矢量数据集：{shp_path1}")
        return False
    layer1 = ds1.GetLayer()
    extent1 = layer1.GetExtent()
    ds2 = ogr.Open(shp_path2)
    if ds2 is None:
        print(f"无法打开矢量数据集：{shp_path2}")
        ds1 = None  # 关闭第一个数据集
        return False
    layer2 = ds2.GetLayer()
    extent2 = layer2.GetExtent()
    # 判断四至范围是否相交
    if extent1[0] < extent2[1] and extent1[1] > extent2[0] and extent1[2] < extent2[3] and extent1[3] > extent2[2]:
        return True
    else:
        return False


def add_polygon(simple_polygon, out_lyr, in_feat, field_defns):
    feat_defn = out_lyr.GetLayerDefn()
    polygon = ogr.CreateGeometryFromWkb(simple_polygon)
    out_feat = ogr.Feature(feat_defn)
    out_feat.SetGeometry(polygon)
    for field_defn in field_defns:
        out_feat.SetField(field_defn.GetName(), in_feat.GetField(field_defn.GetName()))
    out_lyr.CreateFeature(out_feat)


def split_multipolyg2poly(in_shp_path, out_shp_path):
    make_file(os.path.dirname(out_shp_path))
    gdal.SetConfigOption('SHAPE_ENCODING', "GBK")
    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
    gdal.UseExceptions()
    driver = ogr.GetDriverByName('ESRI Shapefile')
    in_ds = driver.Open(in_shp_path, 0)
    in_lyr = in_ds.GetLayer()
    if os.path.exists(out_shp_path):
        driver.DeleteDataSource(out_shp_path)
    out_ds = driver.CreateDataSource(out_shp_path)
    time.sleep(0.1)
    out_lyr = out_ds.CreateLayer(out_shp_path, srs=in_lyr.GetSpatialRef(), geom_type=ogr.wkbPolygon)
    defns = in_lyr.GetLayerDefn()
    FieldDefns = []
    for i in range(defns.GetFieldCount()):
        out_lyr.CreateField(defns.GetFieldDefn(i))
        FieldDefns.append(defns.GetFieldDefn(i))

    for in_feat in in_lyr:
        geom = in_feat.GetGeometryRef()
        if geom.GetGeometryName() == "MULTIPOLYGON":
            for geom_part in geom:
                add_polygon(geom_part.ExportToWkb(), out_lyr, in_feat, FieldDefns)
        else:
            add_polygon(geom.ExportToWkb(), out_lyr, in_feat, FieldDefns)

    out_ds.Destroy()
    in_ds.Destroy()


def merge_same_features(shp_path, out_path, attribute='FileName', sort_type=None):
    # 打开输入矢量文件
    input_ds = ogr.Open(shp_path)
    if input_ds is None:
        raise KeyError("无法打开输入矢量文件。")
    input_lyr = input_ds.GetLayer()
    output_driver = ogr.GetDriverByName("ESRI Shapefile")
    make_file(os.path.dirname(out_path))
    output_ds = output_driver.CreateDataSource(out_path)
    if output_ds is None:
        raise KeyError("无法创建输出矢量文件。")
    time.sleep(0.1)
    output_lyr = output_ds.CreateLayer("merged", input_lyr.GetSpatialRef(), ogr.wkbUnknown)
    first_feature = input_lyr[0]
    field_defn = first_feature.GetDefnRef()
    for i in range(field_defn.GetFieldCount()):
        field_def = field_defn.GetFieldDefn(i)
        output_lyr.CreateField(field_def)
    feat_lst = []
    for feat in input_lyr:
        feat_lst.append([feat, feat.GetField(attribute)])
    sorted_feat_lst = sorted(feat_lst, key=lambda x: x[1])
    attr0, feat0, union_geom = None, None, None
    finished = []
    for i, (feat, attr) in enumerate(sorted_feat_lst):
        if attr != attr0:
            attr0, feat0 = attr, feat
            union_geom = feat0.GetGeometryRef()
        else:
            union_geom = union_geom.Union(feat.GetGeometryRef())
        if i == len(sorted_feat_lst) - 1 or sorted_feat_lst[i + 1][1] != attr:
            output_feat = ogr.Feature(output_lyr.GetLayerDefn())
            output_feat.SetGeometry(union_geom)
            for j in range(field_defn.GetFieldCount()):
                output_feat.SetField(j, feat0.GetField(j))
            # output_lyr.CreateFeature(output_feat)
            finished.append(output_feat)
            attr0 = feat0 = union_geom = None
    if sort_type == 'date':
        s_list = sorted(finished, key=lambda x:x.GetField("PbandDate"), reverse=True)
    elif sort_type == 'res':
        s_list = sorted(finished, key=lambda x:x.GetField("Resolution"))
    else:
        s_list = finished
    for feat in s_list:
        output_lyr.CreateFeature(feat)
    del output_ds, input_ds
    return out_path


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


def get_field_in_shp1feat(shp_path, field):
    datasets = ogr.Open(shp_path, 0)
    for layer in datasets:
        if layer.FindFieldIndex(field, True) > 0:
            for feat in layer:
                return feat.GetField(field)
        else:
            raise KeyError('找不到:{}'.format(field))


def get_field_in_shp_all_feat(shp_path, field):
    field_value_list = []
    datasets = ogr.Open(shp_path, 0)
    for layer in datasets:
        if layer.FindFieldIndex(field, True) > 0:
            for feat in layer:
                field_value_list.append(feat.GetField(field))
        else:
            raise KeyError('找不到:{}'.format(field))
    return field_value_list


def is_vector_empty(filename):
    """
    判断矢量数据是否为空。
    参数：
        filename：矢量文件名，字符串类型。
    返回值：
        如果矢量数据为空，则返回True；否则，返回False。
    """
    # 打开矢量文件
    driver = ogr.GetDriverByName('ESRI Shapefile')
    datasource = driver.Open(filename, 0)

    # 判断矢量数据是否为空
    if datasource is None:
        return True
    layer = datasource.GetLayer()
    if layer is None:
        return True
    feat_num = layer.GetFeatureCount()
    if feat_num == 0:
        return True

    del datasource
    return False


def merge_features(shp_path, out_path):
    # 打开输入矢量文件
    input_ds = ogr.Open(shp_path)
    if input_ds is None:
        print("无法打开输入矢量文件。")
        return

    # 获取输入图层
    input_lyr = input_ds.GetLayer()

    # 创建输出矢量文件
    output_driver = ogr.GetDriverByName("ESRI Shapefile")
    output_ds = output_driver.CreateDataSource(out_path)
    if output_ds is None:
        print("无法创建输出矢量文件。")
        return

    # 创建输出图层
    time.sleep(0.1)
    output_lyr = output_ds.CreateLayer("merged", input_lyr.GetSpatialRef(), ogr.wkbUnknown)

    # 获取第一个feature的属性定义
    first_feature = input_lyr[0]
    field_defn = first_feature.GetDefnRef()

    # 创建输出图层的属性表结构
    for i in range(field_defn.GetFieldCount()):
        field_def = field_defn.GetFieldDefn(i)
        output_lyr.CreateField(field_def)

    # 获取所有输入feature的几何对象，并进行Union操作
    union_geom = None
    for feature in input_lyr:
        geom = feature.GetGeometryRef()
        if union_geom is None:
            union_geom = geom.Clone()
        else:
            union_geom = union_geom.Union(geom)

    # 创建新的feature并将其写入到输出图层中
    output_feat = ogr.Feature(output_lyr.GetLayerDefn())
    output_feat.SetGeometry(union_geom)

    # 将第一个feature的属性值赋给新的feature
    for i in range(field_defn.GetFieldCount()):
        output_feat.SetField(i, first_feature.GetField(i))

    output_lyr.CreateFeature(output_feat)

    # 释放资源
    del output_ds
    del input_ds
    return out_path


# 生成结合表
def easy_union_shp(shp_path_list, out_dir=None, output_path=None):
    if output_path:
        make_file(os.path.dirname(output_path))
        out_path = output_path
    elif out_dir:
        basename = os.path.basename(shp_path_list[-1])
        make_file(out_dir)
        out_path = os.path.join(out_dir, basename)
    else:
        raise KeyError('out error')

    if len(shp_path_list) == 0:
        return None
    elif len(shp_path_list) == 1:
        shutil.copy2(shp_path_list[0], output_path)
        shutil.copy2(shp_path_list[0].replace('shp', 'dbf'), output_path.replace('shp', 'dbf'))
        shutil.copy2(shp_path_list[0].replace('shp', 'prj'), output_path.replace('shp', 'prj'))
        shutil.copy2(shp_path_list[0].replace('shp', 'shx'), output_path.replace('shp', 'shx'))
    else:
        pass
    driver = ogr.GetDriverByName('ESRI Shapefile')
    dataset0 = driver.Open(shp_path_list[0])
    layer0 = dataset0.GetLayerByIndex(0)
    srs0 = layer0.GetSpatialRef()
    defn0 = layer0.GetLayerDefn()

    outds = driver.CreateDataSource(out_path)
    time.sleep(0.1)
    outlayer = outds.CreateLayer(out_path, srs=srs0, geom_type=ogr.wkbPolygon)

    FieldDefns = []
    for field_n in range(defn0.GetFieldCount()):
        FieldDefns.append(defn0.GetFieldDefn(field_n))
        outlayer.CreateField(defn0.GetFieldDefn(field_n))

    for i, path in enumerate(shp_path_list):
        shp = driver.Open(path)
        layer = shp.GetLayerByIndex(0)
        count = layer.GetFeatureCount()
        if count == 0:
            continue
        layer.ResetReading()
        for feature in layer:
            geom = feature.GetGeometryRef()
            out_feat = ogr.Feature(outlayer.GetLayerDefn())
            out_feat.SetGeometry(geom)
            for FieldDefn in FieldDefns:
                out_feat.SetField(FieldDefn.GetName(), feature.GetField(FieldDefn.GetName()))
            outlayer.CreateFeature(out_feat)
        shp.Destroy()
    dataset0.Destroy()
    outds.Destroy()
    return out_path


def clip_vector(source_path, clip_path, output_path):
    # 打开源矢量和裁剪矢量
    source_ds = ogr.Open(source_path)
    clip_ds = ogr.Open(clip_path)

    # 获取裁剪几何体
    clip_layer = clip_ds.GetLayer()
    clip_feature = clip_layer.GetNextFeature()
    clip_geometry = clip_feature.GetGeometryRef()

    # 获取源矢量的空间参考
    source_layer = source_ds.GetLayer()
    source_srs = source_layer.GetSpatialRef()

    # 创建输出数据源和图层
    output_ds = ogr.GetDriverByName("ESRI Shapefile").CreateDataSource(output_path)
    time.sleep(0.1)
    output_layer = output_ds.CreateLayer("clipped", srs=source_srs, geom_type=ogr.wkbPolygon)

    # 定义输出图层属性
    source_layer = source_ds.GetLayer()
    for field in source_layer.schema:
        output_layer.CreateField(field)

    # 遍历源图层的要素进行裁剪
    for source_feature in source_layer:
        source_geometry = source_feature.GetGeometryRef()

        # 使用裁剪几何体进行裁剪
        clipped_geometry = source_geometry.Intersection(clip_geometry)

        # 如果裁剪后的几何体不为空，则写入输出图层
        if clipped_geometry:
            clipped_feature = ogr.Feature(output_layer.GetLayerDefn())
            clipped_feature.SetGeometry(clipped_geometry)

            # 设置属性值
            for i in range(source_feature.GetFieldCount()):
                field_value = source_feature.GetField(i)
                clipped_feature.SetField(i, field_value)

            output_layer.CreateFeature(clipped_feature)

    # 关闭数据源

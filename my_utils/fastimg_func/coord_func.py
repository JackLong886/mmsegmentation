from osgeo import osr
import numpy as np


def coord_ras2geo(im_geotrans, coord):
    x0, x_res, _, y0, _, y_res = im_geotrans
    x = x0 + x_res * coord[0]
    y = y0 + y_res * coord[1]
    return x, y


def coord_geo2ras(im_geotrans, coord):
    x0, x_res, _, y0, _, y_res = im_geotrans
    x = int((coord[0] - x0) / x_res + 0.5)
    y = int((coord[1] - y0) / y_res + 0.5)
    return x, y


def coord_ras2ras(im_geotrans_src, im_geotrans_dst, coord):
    x_geo, y_geo = coord_ras2geo(im_geotrans_src, coord)
    return coord_geo2ras(im_geotrans_dst, (x_geo, y_geo))

def get_srs_pair(im_proj):
    '''
    获得给定数据的投影参考系和地理参考系
    :param dataset: GDAL地理数据
    :return: 投影参考系和地理参考系
    '''
    prosrs = osr.SpatialReference()
    prosrs.ImportFromWkt(im_proj)
    geosrs = prosrs.CloneGeogCS()
    return prosrs, geosrs


def proj2geo(im_proj, x, y):
    """
    将投影坐标转为经纬度坐标（具体的投影坐标系由给定数据确定）
    :param dataset: GDAL地理数据
    :param x: 投影坐标x
    :param y: 投影坐标y
    :return: 投影坐标(x, y)对应的经纬度坐标(lon, lat)
    """
    if "PROJCS" in im_proj:
        prosrs, geosrs = get_srs_pair(im_proj)
        ct = osr.CoordinateTransformation(prosrs, geosrs)
        if geosrs.EPSGTreatsAsLatLong():
            return ct.TransformPoint(x, y)[:2][::-1]
        else:
            return ct.TransformPoint(x, y)[:2]


def geo2proj(im_proj, lon, lat):
    """
    将经纬度坐标转为投影坐标（具体的投影坐标系由给定数据确定）
    :param dataset: GDAL地理数据
    :param lon: 地理坐标lon经度
    :param lat: 地理坐标lat纬度
    :return: 经纬度坐标(lon, lat)对应的投影坐标
    """
    if "PROJCS" in im_proj:
        prosrs, geosrs = get_srs_pair(im_proj)
        ct = osr.CoordinateTransformation(geosrs, prosrs)
        if prosrs.EPSGTreatsAsNorthingEasting():
            return ct.TransformPoint(lat, lon)[:2][::-1]
        else:
            return ct.TransformPoint(lat, lon)[:2]


def cr2geo(im_proj, geo_trans, col, row):
    '''
    根据GDAL的六参数模型将影像图上坐标（行列号）转为投影坐标或地理坐标（根据具体数据的坐标系统转换）
    :param geo_trans: GDAL地理数据的六参数模型
    :param row: 像素的行号
    :param col: 像素的列号
    :return: 行列号(row, col)对应的投影坐标或地理坐标(x, y)
    '''
    px = geo_trans[0] + col * geo_trans[1] + row * geo_trans[2]  # 注意col和row的顺序
    py = geo_trans[3] + col * geo_trans[4] + row * geo_trans[5]
    if "PROJCS" in im_proj:
        lon, lat = proj2geo(im_proj, px, py)
        return lon, lat
    return px, py


def geo2cr(im_proj, geo_trans, lon, lat):
    '''
    根据GDAL的六参数模型将给定的投影或地理坐标转为影像图上坐标（行列号）
    :param geo_trans: GDAL地理数据的六参数模型
    :param x: 投影或地理坐标x
    :param y: 投影或地理坐标y
    :return: 投影坐标或地理坐标(x, y)对应的影像图上行列号(row, col)
    '''
    if "PROJCS" in im_proj:
        x, y = geo2proj(im_proj, lon, lat)
    else:
        x, y = lon, lat
    a = np.array([[geo_trans[1], geo_trans[2]], [geo_trans[4], geo_trans[5]]])
    b = np.array([x - geo_trans[0], y - geo_trans[3]])
    col, row = np.linalg.solve(a, b)  # 返回值顺序调整为col, row
    return int(round(col)), int(round(row))  # 使用round改进精度处理


def cr2cr(col, row, src_proj, src_geotrans, dst_proj, dst_geotrans):
    lon, lat = cr2geo(src_proj, src_geotrans, col, row)
    return geo2cr(dst_proj, dst_geotrans, lon, lat)


def ext2ext(extent, in_im_geotrans, out_im_geotrans):
    geo_pct = coord_ras2geo(in_im_geotrans, (extent[0], extent[1]))
    out_ras_pct = coord_geo2ras(out_im_geotrans, geo_pct)
    return [out_ras_pct[0], out_ras_pct[1], extent[2], extent[3]]

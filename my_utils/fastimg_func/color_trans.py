import multiprocessing as mp
import os
import shutil
from functools import partial
from osgeo import gdal
import cv2
import numpy as np

from . import color_transfer_para, generate_overview, image_stats_path, IMAGE3, make_file, program_progress


def img_clahe(img_path, out_img_path, clip_limit=1, tiles_grid_size=(64, 64)):
    clahe = cv2.createCLAHE()
    clahe.setClipLimit(clip_limit)
    clahe.setTilesGridSize(tiles_grid_size)
    image = IMAGE3(img_path)
    image.create_img(out_img_path)
    for i in range(image.im_bands):
        band_data = image.get_extent(bands=i)
        out_data = clahe.apply(band_data)
        out_data[band_data == 0] = 0
        image.write_extent(out_data, bands=i)
    del image


def transfer_lhm(content, reference, content_mask=None, reference_mask=None):
    """Transfers colors from a reference image to a content image using the
    Linear Histogram Matching.

    content: NumPy array (HxWxC)
    reference: NumPy array (HxWxC)
    """
    # Convert HxWxC image to a (H*W)xC matrix.
    shape = content.shape
    assert len(shape) == 3
    content_m_r = content_mask.reshape(-1)
    reference_m_r = reference_mask.reshape(-1)
    content = content.reshape(-1, shape[-1]).astype(np.float32)
    reference = reference.reshape(-1, shape[-1]).astype(np.float32)

    def matrix_sqrt(X):
        eig_val, eig_vec = np.linalg.eig(X)
        return eig_vec.dot(np.diag(np.sqrt(eig_val))).dot(eig_vec.T)

    mu_content = np.mean(content[content_m_r > 0, :], axis=0)
    mu_reference = np.mean(reference[reference_m_r > 0, :], axis=0)

    cov_content = np.cov(content[content_m_r > 0, :], rowvar=False)
    cov_reference = np.cov(reference[reference_m_r > 0, :], rowvar=False)

    # Add a small regularization term to the covariance matrices
    epsilon = 1e-5
    cov_content = cov_content + epsilon * np.eye(cov_content.shape[0])
    cov_reference = cov_reference + epsilon * np.eye(cov_reference.shape[0])

    result = matrix_sqrt(cov_reference)
    result = result.dot(np.linalg.inv(matrix_sqrt(cov_content)))
    result = result.dot((content - mu_content).T).T
    result = result + mu_reference
    # Restore image dimensions.
    result = result.reshape(shape).clip(1, 255).round().astype(np.uint8)
    result[content_mask == 0, :] = 0
    return result


def write_extent_mp(img: IMAGE3, extent, imdata):
    img.write_extent(imdata, extent)


def process_extent_mp(ref_stat, img_path, src_stat, extent, to_print):
    print(to_print)
    return color_transfer_para(imdata=IMAGE3(img_path).get_extent(extent),
                               img_stats=src_stat, ref_stats=ref_stat,
                               clip=True, preserve_paper=False)


class ColorTransformer:
    def __init__(self, ref_path, pct=1, work_dir='workspace', del_cache=True, win_size=(2048, 2048)):
        self.win_size_x, self.win_size_y = win_size
        self.ref_path = ref_path
        self.pct = pct
        self.work_dir = os.path.join(work_dir, "ColorTransformer")
        self.del_cache = del_cache
        self.ref_stat = image_stats_path(generate_overview(ref_path, self.work_dir, pct=self.pct, frmt='VRT'))
        make_file(work_dir)

    def __call__(self, src_path, dst_path, tmp_dst_path=None, call_back=None):
        if src_path == self.ref_path:
            return src_path
        if not call_back:
            call_back = program_progress('color_trans', 1, 1)
        src_stat = image_stats_path(generate_overview(src_path, self.work_dir, pct=self.pct, frmt='VRT'))
        src = IMAGE3(src_path)
        tmp_dst_path = os.path.join(self.work_dir, os.path.basename(dst_path)) if tmp_dst_path is None else tmp_dst_path
        make_file(os.path.dirname(tmp_dst_path))
        src.create_img(tmp_dst_path)
        if src.im_height * src.im_width > self.win_size_x * self.win_size_y * 8:
            exts = src.gen_extents(self.win_size_x, self.win_size_y)
            pool = mp.Pool(processes=mp.cpu_count())
            for i, ext in enumerate(exts):
                callback_function = partial(write_extent_mp, src, ext)
                # imdata = src.get_extent(ext)
                # outdata = color_transfer_para(imdata=imdata, img_stats=src_stat, ref_stats=self.ref_stat, clip=True,
                #                               preserve_paper=False)
                # callback_function(outdata)
                to_print = call_back.get2print(i / len(exts))
                args = (self.ref_stat, src_path, src_stat, ext, to_print)
                pool.apply_async(process_extent_mp, args=args, callback=callback_function)
            # 关闭进程池
            pool.close()
            pool.join()
        else:
            imdata = src.get_extent()
            outdata = color_transfer_para(imdata=imdata, img_stats=src_stat, ref_stats=self.ref_stat, clip=False,
                                          preserve_paper=False)
            src.write_extent(outdata)
        img_clahe(tmp_dst_path, dst_path, clip_limit=1, tiles_grid_size=(64, 64))
        return dst_path

    def __del__(self):
        if self.del_cache:
            shutil.rmtree(self.work_dir)


if __name__ == '__main__':
    src_path_list = [
        r"D:\LCJ\imdata\laos\cr\rem_172.tif",
        r"D:\LCJ\imdata\laos\cr\rem_173.tif"
    ]
    ref_path = r"D:\LCJ\imdata\laos\cr\rem_ref_img.tif"
    dst_dir = r"D:\LCJ\imdata\laos\color_trans"
    work_dir = r'D:\LCJ\tmp'
    ct = ColorTransformer(ref_path, pct=1, work_dir=work_dir, win_size=(2048, 2048))
    for src_path in src_path_list:
        dst_path = os.path.join(dst_dir, os.path.basename(src_path).replace('.tif', '_color_trans.tif'))
        ct(src_path, dst_path, None)

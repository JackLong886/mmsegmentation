import os
import secrets
import sys

import cv2
import kornia.feature as KF
import numpy as np
from osgeo import gdal

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                             r"envs\infer\Lib\site-packages\torch\lib"))
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from fastimg_func import Timer, iqr_mean
from fastimg_func import IMAGE3, minmax_stretch2range, percentile_stretch2range
from fastimg_func import coord_geo2ras, coord_ras2geo, find_extent_4match, coord_ras2ras, get_cellsize
from fastimg_func import program_progress, make_file, path2frmt, reproj_img2ref
from DOM_Process.pre_utils.match_utils import draw_match_pts, filter_matches
from lightglue import LightGlue, SuperPoint, rbd
import torch
from torchvision import transforms
import random

import matplotlib
from matplotlib import pyplot as plt

matplotlib.use('QT5Agg')

class ImgMatcher:
    def __init__(self, work_dir, x_winsize=1024, y_winsize=1024, if_BuildOverviews=False, device='cuda:0',
                 match_method='light_glue', get_gcps_message='get_gcps', warp_by_gcp_message='warp_by_gcp',
                 min_win_num=10,
                 min_num_in1win=10, choose_num_in1win=10, reproj_threshold=3., confidence=0.99999, max_iters=50000):
        """
        Args:
            work_dir:
            x_winsize:
            y_winsize:
            min_win_num:最少的有效窗口数量
            min_num_in1win:每个窗口至少包含匹配点数量
            if_BuildOverviews:True, False
            device:cpu, gpu
            match_method: loftr, disk, super_glue, light_glue
            get_gcps_message:
            warp_by_gcp_message:

        """
        self.timer = Timer()
        self.work_dir = work_dir
        make_file(work_dir)
        self.x_winsize = x_winsize
        self.y_winsize = y_winsize
        self.if_BuildOverviews = if_BuildOverviews
        self.device = device
        if not torch.cuda.is_available():
            self.device = 'cpu'
        self.match_method = match_method
        self.choose_num_in1win = choose_num_in1win
        self.min_win_num = min_win_num
        self.min_num_in1win = min_num_in1win
        self.get_gcps_message = get_gcps_message
        self.warp_by_gcp_message = warp_by_gcp_message
        self.valid_win_num = None
        self.if_stretch_ref = None
        self.if_stretch_src = None
        self.ref = None
        self.src = None
        self.reproj_threshold = reproj_threshold
        self.confidence = confidence
        self.max_iters = max_iters
        if self.match_method in ['loftr']:
            self.loftr_matcher = KF.LoFTR(pretrained=os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                                                  'checkpoints/loftr_outdoor.ckpt')).to(self.device)
        elif self.match_method in ['light_glue']:
            # 'disk' 用不了
            self.lg_extractor = SuperPoint(max_num_keypoints=2048).eval().to(self.device)
            self.lg_matcher = LightGlue(features='superpoint').eval().to(self.device)

    def prepare_ref_src(self, src_image_path, ref_image_path, resample_method=gdal.GRA_Bilinear):
        ref_warp_path = os.path.join(self.work_dir, 'ref_warp{}.vrt'.format(secrets.token_hex(4)))
        src_warp_path = os.path.join(self.work_dir, 'src_warp{}.vrt'.format(secrets.token_hex(4)))
        src_reproj_path = os.path.join(self.work_dir, 'src_reproj{}.vrt'.format(secrets.token_hex(4)))
        find_extent_4match(ref_image_path, src_image_path, ref_warp_path, src_warp_path, resample_method)
        src_reproj_path = reproj_img2ref(ref_image_path, src_image_path, src_reproj_path)
        return ref_warp_path, src_warp_path, src_reproj_path

    @staticmethod
    def prepare_data(img: IMAGE3, extent, if_stretch=False, stat=None):
        im_data = img.get_extent(extent)
        if im_data is None:
            return None
        if img.im_bands > 1:
            im_data = im_data.transpose(1, 2, 0)
        in_range = stat[0] if img.im_bands == 1 else stat
        im_data = minmax_stretch2range(im_data, out_range=(0, 255), in_range=in_range) if if_stretch else im_data
        # # 测试，不使用全局参数进行拉伸
        # im_data = minmax_stretch2range(im_data, out_range=(0, 255))

        if img.im_bands == 1:  # gray
            pass
        elif img.im_bands == 3:  # RGB2L
            im_data = cv2.split(cv2.cvtColor(im_data.astype(np.uint8), cv2.COLOR_RGB2LAB))[0]
        elif img.im_bands > 3:  # BGR...2L
            im_data = cv2.split(cv2.cvtColor(im_data[..., :3].astype(np.uint8), cv2.COLOR_RGB2LAB))[0]
        else:  # select first channels
            im_data = im_data[..., 0]
        im_data = percentile_stretch2range(im_data, 2, 99, (0, 255)).astype(np.uint8)
        return im_data

    def pcts_local2global(self, extent_lst, mkpts_src_lst, mkpts_ref_lst):
        global_mkpts_src_lst = np.empty((0, 2), dtype=np.float32)
        global_mkpts_ref_lst = np.empty((0, 2), dtype=np.float32)
        for extent, mkpts_src, mkpts_ref in zip(extent_lst, mkpts_src_lst, mkpts_ref_lst):
            for p1, p2 in zip(mkpts_src, mkpts_ref):
                p1[0], p1[1] = p1[0] + self.ext2ext(extent)[0], p1[1] + self.ext2ext(extent)[1]
                p2[0], p2[1] = p2[0] + extent[0], p2[1] + extent[1]
                global_mkpts_src_lst = np.append(global_mkpts_src_lst, [p1], axis=0)
                global_mkpts_ref_lst = np.append(global_mkpts_ref_lst, [p2], axis=0)
        return global_mkpts_src_lst, global_mkpts_ref_lst

    def pcts_ransac(self, mkpts_src_lst, mkpts_ref_lst, ransac_method=cv2.USAC_MAGSAC):
        if mkpts_src_lst.shape[0] < 10:
            raise KeyError(f'无法找到足够的同名匹配点：{mkpts_src_lst.shape[0]}')
        # cv2.USAC_MAGSAC
        H, inliers = cv2.findHomography(mkpts_src_lst, mkpts_ref_lst, ransac_method, self.reproj_threshold,
                                        maxIters=self.max_iters, confidence=self.confidence)
        use_inliers = inliers > 0
        match_pct_num = len(mkpts_src_lst[inliers[..., 0]])
        if match_pct_num < 10:
            raise KeyError(f'无法找到足够的同名匹配点：{match_pct_num}')
        print('threshold:{:.4f}, ransac:{} -> {}'.format(self.reproj_threshold, len(mkpts_src_lst),
                                                         len(mkpts_src_lst[use_inliers[..., 0]])))

        mkpts_src_lst, mkpts_ref_lst = mkpts_src_lst[use_inliers[..., 0]], mkpts_ref_lst[use_inliers[..., 0]]
        return mkpts_src_lst, mkpts_ref_lst, H

    @staticmethod
    def pcts2gcps(mkpts_src_lst, mkpts_ref_lst, ref: IMAGE3):
        gcps = []
        for p1, p2 in zip(mkpts_src_lst, mkpts_ref_lst):
            x_geo, y_geo = coord_ras2geo(ref.im_geotrans, (p2[0], p2[1]))
            gcps.append(gdal.GCP(x_geo, y_geo, 0, float(p1[0]), float(p1[1])))
        return gcps

    def get_ext_mkpts(self, extent, draw_dir=None):
        ref_data = self.prepare_data(self.ref, extent, self.if_stretch_ref, self.ref.statis)
        src_data = self.prepare_data(self.src, self.ext2ext(extent), self.if_stretch_src, self.src.statis)
        if ref_data is None or src_data is None or np.max(ref_data) == 0 or np.max(src_data) == 0:
            print('noData')
            return

        # cv2.imshow('ref', ref_data)
        # cv2.imshow('src', src_data)
        # cv2.waitKey()
        if self.match_method in ['loftr']:
            mkpts_src, mkpts_ref = self.get_match_pcts_loftr(imgray1=src_data, imgray2=ref_data)
        elif self.match_method in ['light_glue']:
            mkpts_src, mkpts_ref = self.get_match_pcts_lightglue(imgray1=src_data, imgray2=ref_data)
        elif self.match_method == 'light_glue_onnx':
            mkpts_src, mkpts_ref = self.get_match_pcts_light_glue_onnx(image0=src_data, image1=ref_data)
        else:
            raise KeyError('unkown method: {}'.format(self.match_method))

        if mkpts_ref is None or len(mkpts_ref) < self.min_num_in1win:
            print('point num less than {}, continue...'.format(self.min_num_in1win))
            return
        else:
            # mkpts_src, mkpts_ref = filter_matches(mkpts_src, mkpts_ref, 2., 100.)
            if draw_dir:  # 绘出每个窗口的匹配点
                fname_src = os.path.join(self.work_dir, '{}src.tif'.format(extent))
                fname_ref = os.path.join(self.work_dir, '{}ref.tif'.format(extent))
                draw_path = os.path.join(draw_dir, "{}.tif".format(extent))
                cv2.imwrite(fname_src, src_data)
                cv2.imwrite(fname_ref, ref_data)
                draw_match_pts(fname_src, fname_ref, draw_path, mkpts_src, mkpts_ref,
                               num=50, show_in_win=False, quite=True, thickness=1)

            # """测试：控制每个窗口的匹配点数量"""
            # num_pct = mkpts_src.shape[0]
            # random_indices = np.random.choice(num_pct, size=self.choose_num_in1win, replace=False)
            # mkpts_src, mkpts_ref = mkpts_src[random_indices], mkpts_ref[random_indices]
            return extent, mkpts_src, mkpts_ref

    def ext2ext(self, extent, in_img3: IMAGE3 = None, out_img3: IMAGE3 = None):
        if in_img3 is None:
            in_img3 = self.ref
        if out_img3 is None:
            out_img3 = self.src
        geo_pct = coord_ras2geo(in_img3.im_geotrans, (extent[0], extent[1]))
        col, row = coord_geo2ras(out_img3.im_geotrans, geo_pct)
        col = 0 if col < 0 else col
        row = 0 if row < 0 else row
        if col > out_img3.im_width - extent[2]:
            col = out_img3.im_width - extent[2]
        if row > out_img3.im_width - extent[3]:
            row = out_img3.im_width - extent[3]
        return [col, row, extent[2], extent[3]]

    def select_extent(self, img, extents):
        x_n, y_n = int(img.im_width / self.x_winsize) + 1, int(img.im_height / self.y_winsize) + 1
        select_num_x = max(5, int(img.im_width / 10000))
        select_num_y = max(5, int(img.im_height / 10000))
        if select_num_x * select_num_y >= len(extents):
            return extents
        select_exts = []
        for y in range(0, y_n, int(y_n / select_num_y) + 1):
            y_exts = extents[x_n * y:(1 + y) * x_n]
            se = [y_exts[int(i * len(y_exts) / select_num_x)] for i in range(select_num_x)] + [y_exts[-1]]
            select_exts += se
        y = y_n - 1
        y_exts = extents[x_n * y:(1 + y) * x_n]
        se = [y_exts[int(i * len(y_exts) / select_num_x)] for i in range(select_num_x)] + [y_exts[-1]]
        select_exts += se
        return list(set(select_exts))

    def get_mkpts(self, src_image_path=None, ref_image_path=None, draw_dir=None, call_back=None, select_type='grid'):
        call_back = program_progress(self.get_gcps_message, 1, 1) if call_back is None else call_back
        self.src, self.ref = IMAGE3(src_image_path, if_print=True), IMAGE3(ref_image_path, if_print=True)
        self.if_stretch_src = True if self.src.compute_bit_depth(True) > 255 else False
        self.if_stretch_ref = True if self.ref.compute_bit_depth(True) > 255 else False
        extents = self.ref.gen_extents(self.x_winsize, self.y_winsize)
        ext_len = len(extents)
        if_supplemental = True
        """窗口选择"""
        if select_type == 'grid':
            select_extent = self.select_extent(self.ref, extents)
        elif 'random' in select_type:
            select_win_num = int(select_type.split(':')[1])
            select_idx = np.random.choice(range(1, ext_len), min(select_win_num, ext_len))
            select_extent = [extents[i] for i in select_idx]
            if_supplemental = False
        else:
            raise KeyError(f'incorrect select_type:{select_type}')

        select_ext_len = len(select_extent)
        print('select img patch:{}/{}'.format(select_ext_len, ext_len))
        extent_lst, mkpts_src_lst, mkpts_ref_lst = [], [], []
        valid_win_num = 0
        for i in range(select_ext_len):
            call_back(i / select_ext_len)
            try:
                mkpts_data = self.get_ext_mkpts(select_extent[i], draw_dir)
            except:
                mkpts_data = None
            if mkpts_data is not None:
                extent, mkpts_src, mkpts_ref = mkpts_data
            else:
                print('find no points in extent:{}, valid_win_num:{}'.format(select_extent[i], valid_win_num))
                continue
            extent_lst.append(select_extent[i])
            mkpts_src_lst.append(mkpts_src)
            mkpts_ref_lst.append(mkpts_ref)
            valid_win_num += 1
            print('win:{}/{} , pcts:{} , valid_win:{}'.
                  format(i + 1, len(select_extent), len(mkpts_src), valid_win_num))
        if if_supplemental:
            rest_extent_lst = list(set(extents) - set(select_extent))
            while valid_win_num < self.min_win_num:
                if len(rest_extent_lst) == 0:
                    break
                random_idx = random.randint(0, len(rest_extent_lst) - 1)
                new_extent = rest_extent_lst.pop(random_idx)
                try:
                    extent, mkpts_src, mkpts_ref = self.get_ext_mkpts(new_extent, draw_dir)
                    extent_lst.append(extent)
                    mkpts_src_lst.append(mkpts_src)
                    mkpts_ref_lst.append(mkpts_ref)
                    valid_win_num += 1
                    print('supplemental win :{}, valid_win_num:{}, pct nums:{}'.
                          format(new_extent, valid_win_num, len(mkpts_src)))
                except:
                    print(
                        'find no points in supplemental extent:{}, valid_win_num:{}'.format(new_extent, valid_win_num))
                    continue

        global_mkpts_src_lst, global_mkpts_ref_lst = self.pcts_local2global(extent_lst, mkpts_src_lst, mkpts_ref_lst)
        return global_mkpts_src_lst, global_mkpts_ref_lst

    @staticmethod
    def record(src_image_path, ref_image_path, dst_image_path, record_txt):
        with open(record_txt, 'w') as file:
            file.write('src:{}\n'.format(src_image_path))
            file.write('ref:{}\n'.format(ref_image_path))
            file.write('dst:{}\n'.format(dst_image_path))

    def get_match_pcts_loftr(self, imgray1, imgray2):
        to_tensor = transforms.ToTensor()
        assert len(imgray1.shape) == 2 and len(imgray2.shape) == 2
        img1_gray, img2_gray = to_tensor(imgray1).unsqueeze(0), to_tensor(imgray2).unsqueeze(0)
        input_dict = {"image0": img1_gray.to(self.device), "image1": img2_gray.to(self.device)}
        with torch.inference_mode():
            correspondences = self.loftr_matcher(input_dict)
        mkpts0 = correspondences["keypoints0"].cpu().numpy()
        mkpts1 = correspondences["keypoints1"].cpu().numpy()
        return mkpts0, mkpts1

    def get_match_pcts_light_glue_onnx(self, image0, image1):
        image0 = (image0[np.newaxis, np.newaxis, ...] / 255.).astype(np.float32)
        image1 = (image1[np.newaxis, np.newaxis, ...] / 255.).astype(np.float32)
        try:
            m_kpts0, m_kpts1 = self.light_glue_runner.run(image0, image1, np.asarray([1, 1]), np.asarray([1, 1]))
            return m_kpts0, m_kpts1
        except:
            return [], []

    def get_match_pcts_lightglue(self, imgray1, imgray2):
        """
        Args:
            imgray1: gray img1
            imgray2: gray img2
        Returns: m_kpts1, m_kpts2
        """
        # from triangluted_net.tools import display_images
        # display_images([imgray1, imgray2])
        to_tensor = transforms.ToTensor()
        assert len(imgray1.shape) == 2 and len(imgray2.shape) == 2
        img1_gray, img2_gray = to_tensor(imgray1).unsqueeze(0), to_tensor(imgray2).unsqueeze(0)
        feats1 = self.lg_extractor.extract(img1_gray.to(self.device))
        feats2 = self.lg_extractor.extract(img2_gray.to(self.device))
        if feats1['keypoints'].size(1) and feats2['keypoints'].size(1):
            matches01 = self.lg_matcher({'image0': feats1, 'image1': feats2})
        else:
            return None, None

        feats1, feats2, matches01 = [rbd(x) for x in [feats1, feats2, matches01]]  # remove batch dimension
        kpts0, kpts1, matches = feats1['keypoints'], feats2['keypoints'], matches01['matches']
        m_kpts1, m_kpts2 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
        return m_kpts1.cpu().numpy(), m_kpts2.cpu().numpy()

    def warp_img_by_gcps(self, src_image_path, dst_image_path, gcps, call_back=None, if_BuildOverviews=False):
        print('warp_by_gcp..., gcp num:{}'.format(len(gcps)))
        src = IMAGE3(src_image_path)
        copy_file_name = os.path.join(self.work_dir, 'copy_{}_{}'.
                                      format(secrets.token_hex(4), os.path.splitext(os.path.basename(src.in_file))[0] +
                                             os.path.splitext(os.path.basename(dst_image_path))[1]))
        src.copy_image(copy_file_name)
        src.copy_dataset.SetGCPs(gcps, src.im_proj)
        gdal.SetConfigOption('GDAL_FORCE_CACHING', 'YES')
        warp_options = gdal.WarpOptions(
            resampleAlg=gdal.GRA_Bilinear, multithread=True, callback=call_back, errorThreshold=0.125,
            polynomialOrder=3,
            creationOptions=['TILED=YES', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256'],
            warpOptions=['SKIP_NOSOURCE=YES', 'OPTIMIZE_SIZE=YES', 'GDAL_NUM_THREADS=ALL_CPUS'])
        warp_outds = gdal.Warp(dst_image_path, src.copy_dataset, options=warp_options)
        if not warp_outds:
            raise KeyError('校正失败！')
        else:
            if if_BuildOverviews:
                warp_outds.BuildOverviews('NEAREST', [4, 8, 16, 32, 64, 128])
            warp_outds.FlushCache()
            del src, warp_outds

    def show_mkpts(self, global_mkpts_src_lst, global_mkpts_ref_lst):
        """展示同名点分布"""
        x = []
        y = []
        for srcpt, refpt in zip(global_mkpts_src_lst, global_mkpts_ref_lst):
            x.append(srcpt[0])
            x.append(refpt[0])
            y.append(srcpt[1])
            y.append(refpt[1])
        plt.scatter(x, y)
        plt.xlim(0, self.ref.im_width)
        plt.ylim(0, self.ref.im_height)
        plt.axis('scaled')
        plt.show()
        sys.exit()

    def evaluate_match(self, src_path, ref_path):
        src_warp_path, ref_warp_path, _ = self.prepare_ref_src(src_path, ref_path)
        global_mkpts_src, global_mkpts_ref = self.get_mkpts(src_warp_path, ref_warp_path, select_type='random:10')
        length = len(global_mkpts_src)
        error = []
        for idx in range(length):
            err = np.linalg.norm(global_mkpts_src[idx] - global_mkpts_ref[idx])
            error.append(err)
            # print(global_mkpts_src[idx], global_mkpts_ref[idx], err)
        mean_error = iqr_mean(error)
        print(f'mean error: {mean_error}')
        return mean_error

    def __call__(self, src_image_path, ref_image_path, dst_image_path, draw_dir=None,
                 in_call_back=None):
        if draw_dir:
            make_file(draw_dir)
        dst_dir = os.path.dirname(dst_image_path)
        make_file(dst_dir)
        self.timer.record('prepare work')

        ref_warp_path, src_warp_path, src_reproj_path = self.prepare_ref_src(src_image_path, ref_image_path)
        self.timer.record('prepare_img')
        if in_call_back is None:
            call_back = program_progress(self.get_gcps_message, 1, 2)
        else:
            call_back = in_call_back
            call_back.set_ssubprocess(1, 2)
        global_mkpts_src_lst, global_mkpts_ref_lst = self.get_mkpts(src_warp_path, ref_warp_path, draw_dir, call_back)
        global_mkpts_src_lst, global_mkpts_ref_lst, _ = self.pcts_ransac(global_mkpts_src_lst, global_mkpts_ref_lst)
        """匹配点还原"""
        ratio = get_cellsize(src_warp_path) / get_cellsize(src_reproj_path)
        for i in range(len(global_mkpts_src_lst)):
            x, y = global_mkpts_src_lst[i]
            global_mkpts_src_lst[i] = x * ratio, y * ratio

        # self.show_mkpts(global_mkpts_src_lst, global_mkpts_ref_lst)
        self.gcps = self.pcts2gcps(global_mkpts_src_lst, global_mkpts_ref_lst, self.ref)
        self.timer.record('run win get gcps')
        if in_call_back is None:
            call_back = program_progress(self.get_gcps_message, 2, 2)
        else:
            call_back = in_call_back
            call_back.set_ssubprocess(2, 2)
        self.warp_img_by_gcps(src_reproj_path, dst_image_path, self.gcps, call_back, self.if_BuildOverviews)
        self.timer.record('warp by gcps')
        # self.record(src_image_path, ref_image_path, dst_image_path, record_txt=os.path.join(dst_dir, 'info.txt'))
        call_back.finish_program()
        self.timer.show_time()


if __name__ == '__main__':
    src_image_path = r"D:\TMP\DOM\GF1_PMS1_E110.2_N24.4_20221026_L1A0006852535-PAN1_output.tiff"
    ref_image_path = r"D:\TMP\DOM\GF1B_PMS_E110.2_N24.6_20221017_L1A1228202900-PAN_output.tiff"
    dst_dir = r'D:\TMP\rpc'
    work_dir = r'D:\TMP'

    mat_mathod = 'light_glue'
    method = '{}'.format(mat_mathod)
    x_winsize, y_winsize = 1024, 1024
    base_name = os.path.splitext(os.path.basename(src_image_path))[0]
    dst_image_path = os.path.join(dst_dir, 'regis_{}_x{}y{}_{}.tif'.
                                  format(base_name, x_winsize, y_winsize, method))
    draw_dir = os.path.join(dst_dir, os.path.splitext(os.path.basename(dst_image_path))[0])
    matcher = ImgMatcher(work_dir, x_winsize=x_winsize, y_winsize=y_winsize,
                         if_BuildOverviews=True, device='cuda', match_method=mat_mathod,
                         min_win_num=10, min_num_in1win=10)
    matcher(src_image_path, ref_image_path, dst_image_path, draw_dir=None)

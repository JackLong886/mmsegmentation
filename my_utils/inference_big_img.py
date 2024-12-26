import matplotlib
import numpy as np
import tqdm

from fastimg_func import IMAGE4, pad_win, depad_win
from mmseg.apis import init_model, inference_model

matplotlib.use('TKAGG')


def imdata_gdal2cv(imdata):
    return imdata.transpose(1, 2, 0)[:, :, ::-1]


if __name__ == '__main__':
    config_path = "../configs/segformer/segformer_mit-b0_8xb2-20k_sugar-512x512.py"
    checkpoint_path = "../work_dirs/segformer_mit-b0_8xb2-20k_sugar-512x512/iter_20000.pth"
    img_path = r"D:\Desktop\面试\小平阳裁剪范围\小平阳裁剪范围.jpg"
    out_path = r'D:\Desktop\面试\小平阳裁剪范围\小平阳裁剪范围_甘蔗提取.tif'
    winsize, padding = 1024, 100

    img = IMAGE4(img_path)
    img.create_img(out_path, out_bands=1)
    exts = img.gen_extents(winsize, winsize)
    model = init_model(config_path, checkpoint_path)
    for ext in tqdm.tqdm(exts):
        pad_data = pad_win(img, ext, padding=padding)
        if np.all(pad_data == 0):
            continue
        pad_data = imdata_gdal2cv(pad_data)
        pad_result = inference_model(model, img=pad_data)
        out_data = pad_result.pred_sem_seg.data.squeeze().detach().cpu().numpy()
        depad_result = depad_win(out_data, padding)
        img.write_extent(depad_result, ext)

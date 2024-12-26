import cv2
import matplotlib

from crop_merge4pad_extent import read_tiff
from mmseg.apis import init_model, inference_model

matplotlib.use('TKAGG')
config_path = "../configs/segformer/segformer_mit-b0_8xb2-20k_sugar-512x512.py"
checkpoint_path = "../work_dirs/segformer_mit-b0_8xb2-20k_sugar-512x512/iter_20000.pth"

# 初始化模型并加载权重
model = init_model(config_path, checkpoint_path)

img_path = r"..\data\sugar\img_dir\val\00c2e1fd8dedda55ebbe.tif"
imdata = cv2.imread(img_path)
print(imdata[0, 0])
result = inference_model(model, img=imdata)
out_data = result.pred_sem_seg.data.squeeze().detach().cpu().numpy()
cv2.imwrite('result.png', out_data * 150)

imdata = read_tiff(img_path).transpose(1, 2, 0)[:, :, ::-1]
print(imdata[0, 0])
result = inference_model(model, img=imdata)
out_data = result.pred_sem_seg.data.squeeze().detach().cpu().numpy()
cv2.imwrite('result2.png', out_data * 150)

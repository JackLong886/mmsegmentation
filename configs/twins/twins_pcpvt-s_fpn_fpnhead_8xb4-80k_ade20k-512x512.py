_base_ = [
    '../_base_/models/twins_pcpvt-s_fpn.py', '../_base_/datasets/sugar.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_1k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001),
    clip_grad=None)

_base_ = ['../../_base_/default_runtime.py']

# 모델 설정
model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN',
        gcn_adaptive='init',
        gcn_with_res=True,
        tcn_type='mstcn',
        graph_cfg=dict(layout='coco', mode='spatial'),
        in_channels=2),  # x, y 좌표만 사용
    cls_head=dict(type='GCNHead', num_classes=13, in_channels=256))  # 13개 클래스

# 데이터셋 설정
dataset_type = 'PoseDataset'
ann_file_train = 'data/pkl/train.pkl'
ann_file_val = 'data/pkl/val.pkl'
train_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='UniformSampleFrames', clip_len=100),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),  # 한 명만 처리
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='UniformSampleFrames', clip_len=100, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='UniformSampleFrames', clip_len=100, num_clips=10, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        split='xsub_train'))

val_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        pipeline=val_pipeline,
        split='xsub_val',
        test_mode=True))

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        pipeline=test_pipeline,
        split='xsub_val',
        test_mode=True))

# 평가 설정
val_evaluator = [dict(type='AccMetric')]
test_evaluator = val_evaluator

# 학습 설정
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=50,
    val_begin=1,
    val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 옵티마이저 설정
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        eta_min=0,
        T_max=50,
        by_epoch=True,
        convert_to_iter_based=True)
]

optim_wrapper = dict(
    optimizer=dict(
        type='SGD',
        lr=0.1,
        momentum=0.9,
        weight_decay=0.0005,
        nesterov=True))

# 기본 훅 설정
default_hooks = dict(
    checkpoint=dict(interval=1),
    logger=dict(interval=100))

# 학습률 자동 스케일링 설정
auto_scale_lr = dict(enable=False, base_batch_size=128) 
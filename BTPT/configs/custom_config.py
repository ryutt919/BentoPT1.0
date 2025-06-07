import os

# 기본 경로 설정
base_dir = 'data'
class_name = 'pull_ups'  # 현재 처리 중인 클래스 이름
ann_file_train = os.path.join(base_dir, class_name, 'pkl', 'train.pkl')
ann_file_val = os.path.join(base_dir, class_name, 'pkl', 'val.pkl')

auto_scale_lr = dict(base_batch_size=128, enable=False)
data_root = '.'
dataset_type = 'PoseDataset'
default_hooks = dict(
    checkpoint=dict(interval=1, save_best='auto', type='CheckpointHook'),
    logger=dict(ignore_last=False, interval=1, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    runtime_info=dict(type='RuntimeInfoHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmaction'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=20)
model = dict(
    backbone=dict(
        gcn_adaptive='init',
        gcn_with_res=True,
        graph_cfg=dict(layout='coco', mode='spatial'),
        in_channels=3,
        tcn_type='mstcn',
        type='STGCN'),
    cls_head=dict(in_channels=256, num_classes=9, type='GCNHead'),
    type='RecognizerGCN')
optim_wrapper = dict(
    optimizer=dict(
        lr=0.1, momentum=0.9, nesterov=True, type='SGD', weight_decay=0.0005))
param_scheduler = [
    dict(
        T_max=50,
        by_epoch=True,
        convert_to_iter_based=True,
        eta_min=0,
        type='CosineAnnealingLR'),
]
randomness = dict(deterministic=False, diff_rank_seed=False, seed=None)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file=ann_file_val,
        pipeline=[
            dict(type='PreNormalize2D'),
            dict(dataset='coco', feats=[
                'j',
            ], type='GenSkeFeat'),
            dict(
                clip_len=32,
                num_clips=10,
                test_mode=True,
                type='UniformSampleFrames'),
            dict(type='PoseDecode'),
            dict(num_person=1, type='FormatGCNInput'),
            dict(type='PackActionInputs'),
        ],
        split='xsub_val',
        test_mode=True,
        type='PoseDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = [
    dict(
        type='AccMetric',
        metric_list=('top_k_accuracy', 'mean_class_accuracy'),
        metric_options={
            'top_k_accuracy': {'topk': (1, 3)}  # top-1과 top-3 정확도
        }
    ),
    dict(type='ConfusionMatrix')
]
test_pipeline = [
    dict(type='PreNormalize2D'),
    dict(dataset='coco', feats=[
        'j',
    ], type='GenSkeFeat'),
    dict(
        clip_len=32, num_clips=10, test_mode=True, type='UniformSampleFrames'),
    dict(type='PoseDecode'),
    dict(num_person=1, type='FormatGCNInput'),
    dict(type='PackActionInputs'),
]
train_cfg = dict(
    max_epochs=50, type='EpochBasedTrainLoop', val_begin=1, val_interval=1)
train_dataloader = dict(
    batch_size=16,
    dataset=dict(
        ann_file=ann_file_train,
        pipeline=[
            dict(type='PreNormalize2D'),
            dict(dataset='coco', feats=[
                'j',
            ], type='GenSkeFeat'),
            dict(clip_len=32, type='UniformSampleFrames'),
            dict(type='PoseDecode'),
            dict(num_person=1, type='FormatGCNInput'),
            dict(
                meta_keys=(
                    'frame_dir',
                    'total_frames',
                    'label',
                    'start_index',
                    'modality',
                    'fps',
                ),
                type='PackActionInputs'),
        ],
        split='xsub_train',
        type='PoseDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='PreNormalize2D'),
    dict(dataset='coco', feats=[
        'j',
    ], type='GenSkeFeat'),
    dict(clip_len=32, type='UniformSampleFrames'),
    dict(type='PoseDecode'),
    dict(num_person=1, type='FormatGCNInput'),
    dict(type='PackActionInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=16,
    dataset=dict(
        ann_file=ann_file_val,
        pipeline=[
            dict(type='PreNormalize2D'),
            dict(dataset='coco', feats=[
                'j',
            ], type='GenSkeFeat'),
            dict(
                clip_len=32,
                num_clips=1,
                test_mode=True,
                type='UniformSampleFrames'),
            dict(type='PoseDecode'),
            dict(num_person=1, type='FormatGCNInput'),
            dict(
                meta_keys=(
                    'frame_dir',
                    'total_frames',
                    'label',
                    'start_index',
                    'modality',
                    'fps',
                ),
                type='PackActionInputs'),
        ],
        split='xsub_val',
        test_mode=True,
        type='PoseDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = [
    dict(
        type='AccMetric',
        metric_list=('top_k_accuracy', 'mean_class_accuracy'),
        metric_options={
            'top_k_accuracy': {'topk': (1, 3)}  # top-1과 top-3 정확도
        }
    ),
    dict(type='ConfusionMatrix')
]
val_pipeline = [
    dict(type='PreNormalize2D'),
    dict(dataset='coco', feats=[
        'j',
    ], type='GenSkeFeat'),
    dict(clip_len=32, num_clips=1, test_mode=True, type='UniformSampleFrames'),
    dict(type='PoseDecode'),
    dict(num_person=1, type='FormatGCNInput'),
    dict(type='PackActionInputs'),
]
vis_backends = [
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='ActionVisualizer',
    vis_backends=[
        dict(type='TensorboardVisBackend'),
    ])
# 작업 디렉토리 설정 및 생성

work_dir = os.path.join('models', class_name)
if not os.path.exists(work_dir):
    os.makedirs(work_dir)

# 로그 저장 경로 설정
log_dir = os.path.join(work_dir, 'logs')
vis_dir = os.path.join(work_dir, 'vis_data')

default_hooks.update({
    'logger': dict(
        type='LoggerHook',
        interval=1,
        ignore_last=False    )
})

visualizer.update({
    'vis_backends': [
        dict(
            type='TensorboardVisBackend'        )
    ]
})

# 디렉토리 생성
for d in [work_dir, log_dir, vis_dir]:
    if not os.path.exists(d):
        os.makedirs(d)
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend',
         save_dir='./work_dirs/custom_stgcnpp')
]

visualizer = dict(
    type='ActionVisualizer',
    vis_backends=vis_backends,
    save_dir='./work_dirs/custom_stgcnpp',
    draw_poses=True,          # 포즈 시각화
    draw_feature_maps=True,   # 특징 맵 시각화
    draw_confusion_matrix=True,  # 혼동 행렬 시각화
    save_predictions=True,    # 예측 결과 저장
    fps=10                    # 시각화 프레임 레이트
)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100, ignore_last=False,
               out_dir='./work_dirs/custom_stgcnpp'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='auto'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'),
    visualization=dict(type='VisualizationHook',  # 시각화 훅 추가
                      enable=True,
                      interval=1,
                      save_vis=True)
)

val_evaluator = [
    dict(type='AccMetric'),
    dict(type='ConfusionMatrix',  # 혼동 행렬 평가 추가
         num_classes=13,
         save_path='./work_dirs/custom_stgcnpp/confusion_matrix.png')
]

# 시각화를 위한 추가 설정
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
    visualize=True,  # 시각화 활성화
    save_vis=True    # 시각화 결과 저장
) 
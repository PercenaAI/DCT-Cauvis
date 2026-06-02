import os
import os.path as osp

PROJECT_ROOT = os.getenv('PROJECT_ROOT', '/path/to/project')
OFFICIAL_CKPT = os.getenv(
    'CAUVIS_PRETRAINED',
    osp.join(PROJECT_ROOT, 'pretrained_weights', 'cauvis', 'cauvis_dinohead.pth')
)

_base_ = ['./cauvis_dinov2_dinohead_bs1x4_sdgod_project.py']

model = dict(
    backbone=dict(
        cauvis_config=dict(
            dct_config=dict(
                type='DynamicPromptEvolution',
                hidden_ratio=16,
                scale_init=0.001,
            ),
        ),
    ),
)

load_from = OFFICIAL_CKPT

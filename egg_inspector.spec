# egg_inspector.spec
# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_submodules, collect_data_files
from PySide6.QtCore import QLibraryInfo

entry = "main.py"

# ---- 依赖收集 ----
qt_datas  = collect_data_files('PySide6', include_py_files=True)
qt_hidden = collect_submodules('PySide6')

ultra_hidden = collect_submodules('ultralytics')
timm_hidden  = collect_submodules('timm')
torch_hidden = collect_submodules('torch')
tv_hidden    = collect_submodules('torchvision')
ta_hidden    = collect_submodules('torchaudio')

# 你的模型文件（在根目录）
datas = [
    ('segment.pt', '.'),
    ('ns_ok.pt', '.'),
    ('ok_regressor_hgbr.joblib', '.'),
    ('best_tf_efficientnet_b0_ns_problem_only.pt', '.'),
]

# 只添加 Qt 平台插件/样式（QML 相关不再强制加入）
qt_plugins_dir = QLibraryInfo.path(QLibraryInfo.PluginsPath)

def add_dir_if_exists(src_dir, dst_rel):
    if os.path.isdir(src_dir):
        return [(src_dir, dst_rel)]
    return []

datas += qt_datas
datas += add_dir_if_exists(os.path.join(qt_plugins_dir, 'platforms'), 'PySide6/plugins/platforms')
datas += add_dir_if_exists(os.path.join(qt_plugins_dir, 'styles'),    'PySide6/plugins/styles')

binaries = []
hiddenimports = qt_hidden + ultra_hidden + timm_hidden + torch_hidden + tv_hidden + ta_hidden

block_cipher = None

a = Analysis(
    [entry],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tensorflow','onnx','onnxruntime',
        'torch.backends.cuda','torch.cuda',
        'torch.utils.tensorboard',     # ← 避免 tensorboard 警告
        'torchaudio.prototype',        # ← 可选，减少无关告警
    ],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name='EggInspector',
    debug=False,
    strip=False,
    upx=False,
    console=False,  # 需要黑框日志可改 True
    icon=None
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='EggInspector'
)
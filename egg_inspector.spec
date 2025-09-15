# egg_inspector.spec
# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_submodules, collect_data_files
from PySide6.QtCore import QLibraryInfo

entry = "main.py"

# PySide6 资源
qt_datas  = collect_data_files('PySide6', include_py_files=True)
qt_hidden = collect_submodules('PySide6')

# 其他库隐藏导入
ultra_hidden = collect_submodules('ultralytics')
timm_hidden  = collect_submodules('timm')
torch_hidden = collect_submodules('torch')
tv_hidden    = collect_submodules('torchvision')
ta_hidden    = collect_submodules('torchaudio')

# 模型/权重文件 —— 都在根目录
datas = [
    ('segment.pt', '.'),   # YOLO 模型
    ('ns_ok.pt', '.'),     # OK 分类模型
    ('ok_regressor_hgbr.joblib', '.'),  # 回归模型
    ('best_tf_efficientnet_b0_ns_problem_only.pt', '.'),  # 问题蛋模型
]

# Qt 插件和 QML（有些不用，但打进去更稳）
qt_plugins_dir = QLibraryInfo.path(QLibraryInfo.PluginsPath)
qt_qml_dir     = QLibraryInfo.path(QLibraryInfo.QmlImportsPath)

datas += qt_datas
datas += [
    (os.path.join(qt_plugins_dir, 'platforms'), 'PySide6/plugins/platforms'),
    (os.path.join(qt_plugins_dir, 'styles'),    'PySide6/plugins/styles'),
    (os.path.join(qt_qml_dir, 'QtQuick'),       'PySide6/qml/QtQuick'),
    (os.path.join(qt_qml_dir, 'QtQuick.2'),     'PySide6/qml/QtQuick.2'),
    (os.path.join(qt_qml_dir, 'QtQuick', 'Controls'), 'PySide6/qml/QtQuick/Controls'),
    (os.path.join(qt_qml_dir, 'QtQuick', 'Controls', 'Basic'), 'PySide6/qml/QtQuick/Controls/Basic'),
]

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
    excludes=['tensorflow','onnx','onnxruntime','torch.backends.cuda','torch.cuda'],
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
    console=False,  # 改 True 会有黑框调试日志
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

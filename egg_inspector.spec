# egg_inspector.spec
# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules, collect_data_files
from PyInstaller.utils.hooks.qt import add_qt5_dependencies

entry = "main.py"

# Qt & 依赖
qt_datas = collect_data_files('PySide6', include_py_files=True)
qt_hidden = collect_submodules('PySide6')

# 深度学习依赖（避免隐藏导入）
hiddenimports = []
hiddenimports += qt_hidden
hiddenimports += collect_submodules('torch')
hiddenimports += collect_submodules('torchvision')
hiddenimports += collect_submodules('torchaudio')
hiddenimports += collect_submodules('timm')
hiddenimports += collect_submodules('ultralytics')

# 数据文件：把模型文件复制到可执行文件同目录（'.'）
datas = [
    ('segment.pt', '.'),
    ('ns_ok.pt', '.'),
    ('ok_regressor_hgbr.joblib', '.'),
    ('best_tf_efficientnet_b0_ns_problem_only.pt', '.'),
]
datas += qt_datas

a = Analysis(
    [entry],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tensorflow', 'onnx', 'onnxruntime',
        # 为稳妥不要排 torch._C；如你想减体积再试排除
    ],
    noarchive=False,
)
add_qt5_dependencies(a)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name='EggInspector',
    console=False,   # 需要黑窗就改 True
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name='EggInspector'
)
# -*- coding: utf-8 -*-
"""
Egg Inspector — PySide6 GUI (multi-egg supported, tabbed result)

改版说明：
- 初始仅显示【上传界面】（按钮 + 拖拽提示）。
- 选择图片后切换到【结果页】：
  • 顶部是 Tab（“鸡蛋 1 / 鸡蛋 2 / …”）；
  • 每个 Tab 页面下方是：左侧“该鸡蛋的可视化”、右侧“该鸡蛋的结果表”；
  • 底部常驻“任务日志”，推理进行时显示 loading（不定进度）。

功能：多实例分割/几何特征/OK-问题蛋二分类/合格蛋回归/问题蛋多标签/颜色校正（可选）。
依赖：
  pip install PySide6 opencv-python-headless torch torchvision torchaudio ultralytics timm joblib pillow numpy
"""

import os
import sys
import math
import traceback
from pathlib import Path

import numpy as np
import cv2
import torch
import joblib
import timm
from PIL import Image
import torchvision.transforms as T

from PySide6.QtCore import Qt, QThread, Signal, QSize
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QPlainTextEdit, QSplitter,
    QTableWidget, QTableWidgetItem, QSizePolicy, QMessageBox,
    QTabWidget, QProgressBar, QStackedWidget
)

# ================= 路径与阈值 =================
MODEL_PATH_YOLO      = "segment.pt"  # YOLOv8-seg
CHECKPOINT_OK        = "ns_ok.pt"
# CHECKPOINT_OK        = "runs_okcls/best_tf_efficientnet_b0_ns_ok.pt"
MODEL_NAME_OK        = "tf_efficientnet_b0_ns"
IMG_SIZE_OK          = 384
CHECKPOINT_REG       = "ok_regressor_hgbr.joblib"   # 回归（合格蛋）
# CHECKPOINT_PROBLEM   = "runs_cls_problem_retrain/best_tf_efficientnet_b0_ns_singlelabel_balanced.pt"  # 多标签（问题蛋）
CHECKPOINT_PROBLEM   = "best_tf_efficientnet_b0_ns_problem_only.pt"  # 多标签（问题蛋）

# Epoch 20/20: 100%|██████████| 50/50 [03:00<00:00,  3.61s/it, loss=0.277]
# val ACC=0.8125  macroF1=0.8089
MODEL_NAME_PROBLEM   = "tf_efficientnet_b0_ns"
IMG_SIZE_PROBLEM     = 384

# 多实例：最多数量 & 最小掩膜占比（过滤碎片）
MAX_EGGS = 20
MIN_MASK_AREA_RATIO = 0.001

# 问题蛋标签（需与训练一致）
# 问题蛋标签（默认值；实际以 checkpoint 内保存的 classes 为准）
DEFAULT_DEFECT_COLS = ["斑点蛋","畸形蛋","破壳蛋","软壳蛋","沙皮蛋","小蛋","血蛋","脏蛋","皱皮蛋"]
EXCLUDED_DEFECTS_FOR_EGG_CONF = {"小蛋"}
# 多实例：最多数量 & 最小掩膜占比（过滤碎片）
MAX_EGGS = 20
MIN_MASK_AREA_RATIO = 0.001

# 鸡蛋置信度阈值：若 max(prob_ok, max_defect_prob) < 0.8 -> 丢弃该实例
THRESH_EGG_CONF = 0.2


# ================= 工具函数 =================

def imread_bgr_any(p: str):
    data = np.fromfile(str(p), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    return img


def np_rgb_to_qimage(img_rgb: np.ndarray) -> QImage:
    h, w = img_rgb.shape[:2]
    img_rgb = np.ascontiguousarray(img_rgb)
    return QImage(img_rgb.data, w, h, 3*w, QImage.Format.Format_RGB888).copy()


def overlay_mask_bgr(img_bgr: np.ndarray, mask_u8: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    mask = (mask_u8 > 127).astype(np.uint8)
    color = np.zeros_like(img_bgr)
    color[:, :, 1] = mask * 255
    out = cv2.addWeighted(img_bgr, 1.0 - alpha, color, alpha, 0)
    return out


def fit_ellipse_from_mask(mask_u8: np.ndarray):
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if len(c) < 5:
        return None
    hull = cv2.convexHull(c)
    (cx, cy), (MA, ma), ang = cv2.fitEllipse(hull)
    area = float(cv2.contourArea(hull))
    per = float(cv2.arcLength(hull, True))
    circ = float(4 * math.pi * area / (per ** 2 + 1e-12)) if per > 0 else 0.0
    el_area = math.pi * (MA / 2) * (ma / 2)
    cov = float(area / (el_area + 1e-12))
    return dict(
        center=(cx, cy),
        major_axis_px=max(MA, ma),
        minor_axis_px=min(MA, ma),
        aspect_ratio=max(MA, ma) / (min(MA, ma) + 1e-12),
        angle_deg=ang,
        area_px2=area,
        perimeter_px=per,
        circularity=circ,
        ellipse_cov=cov,
    )


def build_feature_dict_from_geom_and_mask(geom_dict, mask_u8, orig_img_shape):
    H, W = orig_img_shape[:2]
    feat = dict(
        img_w=float(W),
        img_h=float(H),
        area_ratio=float(cv2.countNonZero(mask_u8)) / float(max(1, H * W)),
        major_px=float(geom_dict.get("major_axis_px", np.nan)),
        minor_px=float(geom_dict.get("minor_axis_px", np.nan)),
        aspect_ratio=float(geom_dict.get("aspect_ratio", np.nan)),
        angle_deg=float(geom_dict.get("angle_deg", np.nan)),
        area_px2=float(geom_dict.get("area_px2", np.nan)),
        perimeter_px=float(geom_dict.get("perimeter_px", np.nan)),
        circularity=float(geom_dict.get("circularity", np.nan)),
        ellipse_cov=float(geom_dict.get("ellipse_cov", np.nan)),
    )
    cnts, _ = cv2.findContours((mask_u8 > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        feat.update(dict(bbox_w=float(w), bbox_h=float(h), bbox_aspect=float(max(w, h) / max(1, min(w, h)))))
    return feat


def apply_mask(img_bgr, mask_u8):
    m = (mask_u8 > 127).astype(np.uint8)
    out = img_bgr.copy()
    out[m == 0] = 0
    return out


# ================= 模型管理 =================
class ModelManager:
    def __init__(self, log_fn=None):
        self.log = log_fn or (lambda *a, **k: None)
        self.device_yolo = (
            "mps" if torch.backends.mps.is_available() else (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        )
        self.device_ok = self.device_yolo
        self.device_prob = self.device_yolo

        self.yolo = None
        self.model_ok = None
        self.model_problem = None
        self.reg_pack = None

        # —— 关键：从权重里读到的类别名会存到这里 —— #
        self.defect_cols = list(DEFAULT_DEFECT_COLS)

    def ensure_yolo(self):
        if self.yolo is None:
            from ultralytics import YOLO
            if not Path(MODEL_PATH_YOLO).exists():
                raise FileNotFoundError(f"未找到 YOLO 模型: {MODEL_PATH_YOLO}")
            self.log(f"加载 YOLO 模型: {MODEL_PATH_YOLO} @ {self.device_yolo}")
            self.yolo = YOLO(MODEL_PATH_YOLO)
        return self.yolo

    def ensure_ok(self):
        if self.model_ok is None:
            if not Path(CHECKPOINT_OK).exists():
                raise FileNotFoundError(f"未找到 OK 二分类权重: {CHECKPOINT_OK}")
            self.log(f"加载 OK 模型: {CHECKPOINT_OK} @ {self.device_ok}")
            model = timm.create_model(MODEL_NAME_OK, pretrained=False, num_classes=1)

            # ✅ 仅保留 map_location；兼容两种保存格式
            ckpt = torch.load(CHECKPOINT_OK, map_location=self.device_ok)
            sd = ckpt.get("model", ckpt)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            if missing or unexpected:
                self.log(f"[Warn][OK] missing={len(missing)}, unexpected={len(unexpected)}")

            model.to(self.device_ok).eval()
            self.model_ok = model
        return self.model_ok

    def ensure_reg(self):
        if self.reg_pack is None:
            if not Path(CHECKPOINT_REG).exists():
                raise FileNotFoundError(f"未找到回归模型: {CHECKPOINT_REG}")
            self.log(f"加载回归模型包: {CHECKPOINT_REG}")
            self.reg_pack = joblib.load(CHECKPOINT_REG)
        return self.reg_pack

    def ensure_problem(self):
        """
        从 checkpoint 读取 'classes' 列表，按其长度创建分类头，避免 head 尺寸不匹配。
        """
        if self.model_problem is None:
            if not Path(CHECKPOINT_PROBLEM).exists():
                raise FileNotFoundError(f"未找到问题蛋分类模型: {CHECKPOINT_PROBLEM}")

            self.log(f"加载问题蛋分类模型: {CHECKPOINT_PROBLEM} @ {self.device_prob}")

            # ✅ 仅保留 map_location；兼容两种保存格式
            ckpt = torch.load(CHECKPOINT_PROBLEM, map_location=self.device_prob)

            # 读取类别名；若不存在就用默认
            classes = ckpt.get("classes", None)
            if isinstance(classes, (list, tuple)) and len(classes) >= 2:
                self.defect_cols = list(classes)
            else:
                self.defect_cols = list(DEFAULT_DEFECT_COLS)

            num_classes = len(self.defect_cols)
            model = timm.create_model(MODEL_NAME_PROBLEM, pretrained=False, num_classes=num_classes)

            sd = ckpt.get("model", ckpt)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            if missing or unexpected:
                self.log(f"[Warn][PROB] missing={len(missing)}, unexpected={len(unexpected)}")

            model.to(self.device_prob).eval()
            self.model_problem = model
        return self.model_problem


# ================= 推理线程（多实例+单蛋可视化） =================
class InferenceWorker(QThread):
    progress = Signal(str)
    done = Signal(dict)
    failed = Signal(str)

    def __init__(self, img_path: str, mgr: ModelManager):
        super().__init__()
        self.img_path = img_path
        self.mgr = mgr

    def log(self, msg):
        self.progress.emit(msg)

    def run(self):
        try:
            res = self._run_pipeline(self.img_path)
            self.done.emit(res)
        except Exception:
            self.failed.emit(traceback.format_exc())

    def _run_pipeline(self, img_path: str) -> dict:
        img_bgr = imread_bgr_any(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"无法读取图片: {img_path}")
        H, W = img_bgr.shape[:2]

        # 分割
        self.log("[1/5] YOLO 分割中…")
        yolo = self.mgr.ensure_yolo()
        res = yolo.predict(img_path, imgsz=1280, conf=0.2, iou=0.6, verbose=False)
        r0 = res[0]
        if r0.masks is None or len(r0.masks.data) == 0:
            raise RuntimeError("未检测到掩膜！")

        masks_tensor = r0.masks.data.cpu().numpy()  # [N,h,w] in {0,1}
        confs = r0.boxes.conf.cpu().numpy().tolist() if r0.boxes is not None else [1.0] * masks_tensor.shape[0]
        order = np.argsort(confs)[::-1]

        masks_u8 = []
        kept_indices = []
        for i in order[:MAX_EGGS]:
            m = (masks_tensor[i] * 255).astype(np.uint8)
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            area_ratio = float(cv2.countNonZero(m)) / float(H * W)
            if area_ratio < MIN_MASK_AREA_RATIO:
                continue
            masks_u8.append(m)
            kept_indices.append(i)

        if not masks_u8:
            raise RuntimeError("掩膜过小被过滤或为空，请调低 MIN_MASK_AREA_RATIO。")

        # 预加载模型与变换
        ok_model = self.mgr.ensure_ok()
        val_tfms_ok = T.Compose([
            T.Resize((IMG_SIZE_OK, IMG_SIZE_OK)),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        prob_model = self.mgr.ensure_problem()
        val_tfms_prob = T.Compose([
            T.Resize((IMG_SIZE_PROBLEM, IMG_SIZE_PROBLEM)),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # 颜色校正可用性
        try:
            import egg_color_correct
            has_cc = True
        except Exception:
            has_cc = False
            self.log("颜色校正模块未安装，跳过。")

        eggs = []
        for rank, (i, mask) in enumerate(zip(kept_indices, masks_u8), start=1):
            conf = float(confs[i]) if i < len(confs) else 1.0
            self.log(f"[2/5] 实例 #{rank} 分类中… (conf={conf:.3f})")

            # —— 1) 几何过滤（非蛋形状直接跳过） ——
            geom = fit_ellipse_from_mask(mask)
            # if not geom:
            #     self.log(f"实例 #{rank}: 无法拟合椭圆，丢弃")
            #     continue
            # if not (0.4 <= geom["circularity"] <= 0.95):
            #     self.log(f"实例 #{rank}: circularity={geom['circularity']:.3f} 不像蛋，丢弃")
            #     continue
            # if not (1.0 <= geom["aspect_ratio"] <= 3.0):
            #     self.log(f"实例 #{rank}: aspect_ratio={geom['aspect_ratio']:.2f} 不像蛋，丢弃")
            #     continue
            # if geom["ellipse_cov"] < 0.60:
            #     self.log(f"实例 #{rank}: ellipse_cov={geom['ellipse_cov']:.2f} 太低，丢弃")
            #     continue

            # —— 2) OK 二分类 ——
            img_masked = apply_mask(img_bgr, mask)
            x_ok = val_tfms_ok(Image.fromarray(cv2.cvtColor(img_masked, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(
                self.mgr.device_ok)
            with torch.no_grad():
                prob_ok = torch.sigmoid(ok_model(x_ok)).item()

            # —— 3) 问题蛋多分类（softmax）——
            x_prob = val_tfms_prob(Image.fromarray(cv2.cvtColor(img_masked, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(
                self.mgr.device_prob)
            with torch.no_grad():
                logits = prob_model(x_prob)
                probs_defect = torch.softmax(logits, dim=1).cpu().numpy()[0]  # [C]

            classes = list(self.mgr.defect_cols)
            pred_idx = int(np.argmax(probs_defect))
            pred_name = classes[pred_idx]
            p_defect_max = float(probs_defect[pred_idx])

            # === 仅用于“是不是鸡蛋”的门槛：排除指定类型（如“小蛋”）再取最大 ===
            idx_keep = [i for i, c in enumerate(classes) if c not in EXCLUDED_DEFECTS_FOR_EGG_CONF]
            if idx_keep:
                p_defect_max_excl = float(np.max(probs_defect[idx_keep]))
            else:
                p_defect_max_excl = p_defect_max  # 万一所有类都被排除了，就退化为原来的最大

            # 联合置信度门槛（排除小蛋后的最大问题蛋概率）
            egg_score = max(float(prob_ok), p_defect_max_excl)
            if egg_score < THRESH_EGG_CONF:
                self.log(
                    f"实例 #{rank}: egg_score={egg_score:.3f} < {THRESH_EGG_CONF}，"
                    f"(ok={float(prob_ok):.3f}, max_defect_excl={p_defect_max_excl:.3f}) 判为非蛋，丢弃"
                )
                continue

            # —— 5) 根据 OK 概率确定分支显示（不影响上面的“是不是蛋”判断）——
            label = "合格蛋" if prob_ok >= 0.2 else "问题蛋"

            if label == "合格蛋":
                pack = self.mgr.ensure_reg()
                reg_model = pack["model"];
                feat_names = pack["features"];
                tgt_names = pack["targets"]
                feat_dict = build_feature_dict_from_geom_and_mask(geom or {}, mask, img_bgr.shape)
                X = np.array([[float(feat_dict.get(k, np.nan)) for k in feat_names]], dtype=np.float32)
                y_pred = reg_model.predict(X)[0]
                branch = {"task": "regression", "targets": tgt_names, "values": [float(v) for v in y_pred]}
            else:
                branch = {
                    "task": "multiclass",
                    "classes": classes,
                    "probs": [float(p) for p in probs_defect],
                    "top1": dict(index=pred_idx, name=pred_name, prob=p_defect_max),
                }

            # —— 可视化与收集结果（保持不变）——
            vis_single = overlay_mask_bgr(img_bgr, mask, 0.30)
            if geom:
                cx, cy = map(int, geom["center"])
                ax, ay = int(geom["major_axis_px"] / 2), int(geom["minor_axis_px"] / 2)
                color = (0, 255, 0) if label == "合格蛋" else (0, 0, 255)
                cv2.ellipse(vis_single, (cx, cy), (ax, ay), int(geom["angle_deg"]), 0, 360, color, 2)
                cv2.circle(vis_single, (cx, cy), 5, color, -1)
                tag = f"#{rank} {label}"
                cv2.putText(vis_single, tag, (max(0, cx - 30), max(0, cy - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color,
                            2, cv2.LINE_AA)
            vis_img = np_rgb_to_qimage(cv2.cvtColor(vis_single, cv2.COLOR_BGR2RGB))

            eggs.append({
                "idx": rank,
                "conf": conf,
                "geom": geom,
                "prob_ok": float(prob_ok),
                "label": label,
                "branch": branch,
                "vis_qimage": vis_img,
            })

        self.log(f"[5/5] 共 {len(eggs)} 个实例完成。")
        return {"img_path": img_path, "eggs": eggs}


# ================= 主窗口（上传页 + 结果页[Tab]） =================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Egg Inspector — 多实例分割/分类/回归（Tab 视图）")
        self.resize(1280, 820)
        self.setAcceptDrops(True)

        self.mgr = ModelManager(log_fn=self.log)
        self.worker: InferenceWorker | None = None

        self._build_ui()

    # ---- UI 构建 ----
    def _build_ui(self):
        self.stack = QStackedWidget(self)
        self.setCentralWidget(self.stack)

        # -- Page 1: 上传页 --
        self.page_upload = QWidget()
        up_l = QVBoxLayout(self.page_upload)
        up_l.setContentsMargins(16, 12, 16, 12)

        hint = QLabel("将图片拖拽到此处，或点击下方按钮上传")
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet("QLabel{border:2px dashed #999;color:#666;padding:60px;font-size:16px}")
        self.btn_select1 = QPushButton("选择图片…")
        self.btn_select1.setFixedWidth(160)
        self.btn_select1.clicked.connect(self.on_select)

        up_l.addWidget(hint, stretch=1)
        row = QHBoxLayout(); row.addStretch(1); row.addWidget(self.btn_select1); row.addStretch(1)
        up_l.addLayout(row)
        up_l.addSpacing(20)

        # -- Page 2: 结果页 --
        self.page_result = QWidget()
        rs_l = QVBoxLayout(self.page_result)
        rs_l.setContentsMargins(8, 6, 8, 6)

        # 顶部 busy 指示
        busy_row = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)   # 不定进度
        self.progress.setVisible(False)
        busy_row.addWidget(QLabel("处理进度："))
        busy_row.addWidget(self.progress)
        busy_row.addStretch(1)
        self.btn_back_home = QPushButton("检测其他")
        self.btn_back_home.setToolTip("返回首页，选择另一张图片")
        self.btn_back_home.clicked.connect(self.back_to_home)
        busy_row.addWidget(self.btn_back_home)
        rs_l.addLayout(busy_row)

        # 顶部 Tab（每个鸡蛋一个页面）
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setTabsClosable(False)
        rs_l.addWidget(self.tabs, stretch=5)

        # 任务日志
        rs_l.addWidget(QLabel("任务日志"))
        self.log_box = QPlainTextEdit(); self.log_box.setReadOnly(True); self.log_box.setMaximumBlockCount(1000)
        rs_l.addWidget(self.log_box, stretch=2)

        self.stack.addWidget(self.page_upload)
        self.stack.addWidget(self.page_result)
        self.stack.setCurrentWidget(self.page_upload)

    # ---- 拖拽 ----
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
        else:
            e.ignore()

    def dropEvent(self, e):
        urls = e.mimeData().urls()
        if not urls:
            return
        path = urls[0].toLocalFile()
        if path:
            self.run_infer(path)

    # ---- 选择文件 ----
    def on_select(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.jpg *.jpeg *.png *.bmp)")
        if path:
            self.run_infer(path)

    # ---- 推理入口 ----
    def run_infer(self, path: str):
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "请稍候", "正在处理上一张图片…")
            return
        self.clear_results()
        self.stack.setCurrentWidget(self.page_result)
        self.progress.setVisible(True)
        self.log_box.clear()
        self.log(f"加载图片: {path}")
        self.worker = InferenceWorker(path, self.mgr)
        self.worker.progress.connect(self.log)
        self.worker.done.connect(self.on_done)
        self.worker.failed.connect(self.on_failed)
        self.worker.start()

    def back_to_home(self):
        # 若任务仍在进行，提示一下；依然切回首页（不可开始新任务，直到当前结束）
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "请稍候", "当前仍在处理图片；完成后再选择新图片。")
        self.progress.setVisible(False)
        self.clear_results()
        self.stack.setCurrentWidget(self.page_upload)

    # ---- 结果 Tab 构建 ----
    def build_tab_for_egg(self, eg: dict):
        page = QWidget()
        vbox = QVBoxLayout(page)
        split = QSplitter()
        split.setChildrenCollapsible(False)

        # 左图
        img_area = QWidget()
        img_l = QVBoxLayout(img_area)
        lbl = QLabel()
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("QLabel{border:1px solid #ddd;background:#fafafa}")
        lbl.setMinimumSize(QSize(420, 360))
        if eg.get("vis_qimage"):
            self.set_pix_to_label(lbl, eg["vis_qimage"])
        img_l.addWidget(lbl)

        # 右表
        tbl = QTableWidget(0, 2)
        tbl.setHorizontalHeaderLabels(["项目", "值"])
        tbl.horizontalHeader().setStretchLastSection(True)

        rows = []
        rows.append(("OK 概率", f"{eg.get('prob_ok', float('nan')):.4f}"))
        rows.append(("预测标签", eg.get("label", "")))
        if eg.get("egg_rgb_corr") is not None:
            rows.append(("颜色校正 RGB", str(eg["egg_rgb_corr"])) )
        geom = eg.get("geom") or {}
        if geom:
            rows.append(("—— 几何特征 ——", ""))
            for k, val in geom.items():
                rows.append((k, f"{val:.2f}" if isinstance(val, float) else str(val)))
        br = eg.get("branch", {})
        if br.get("task") == "regression":
            rows.append(("—— 回归(合格蛋) ——", ""))
            for t, val in zip(br.get("targets", []), br.get("values", [])):
                rows.append((t, f"{val:.2f}"))
        elif br.get("task") == "multilabel":
            rows.append(("—— 多标签(问题蛋) ——", ""))
            classes = br.get("classes", [])
            probs = br.get("probs", [])
            for c, p in sorted(zip(classes, probs), key=lambda x: -x[1]):
                rows.append((c, f"{p:.3f}"))
            pos = [c for c, p in zip(classes, probs) if p >= 0.6]
            rows.append(("缺陷判定(>=0.6)", ", ".join(pos) if pos else "无"))
        elif br.get("task") == "multiclass":
            rows.append(("—— 多分类(问题蛋) ——", ""))
            classes = br.get("classes", [])
            probs = br.get("probs", [])
            for c, p in sorted(zip(classes, probs), key=lambda x: -x[1]):
                rows.append((c, f"{p:.3f}"))
            top1 = br.get("top1", {})
            rows.append(("Top-1", f"{top1.get('name', '')} ({top1.get('prob', 0):.3f})"))

        tbl.setRowCount(len(rows))
        for i, (k, val) in enumerate(rows):
            ki, vi = QTableWidgetItem(str(k)), QTableWidgetItem(str(val))
            if "——" in str(k):
                f = ki.font(); f.setBold(True); ki.setFont(f); vi.setFont(f)
            tbl.setItem(i, 0, ki)
            tbl.setItem(i, 1, vi)
        tbl.resizeRowsToContents()

        split.addWidget(img_area)
        split.addWidget(tbl)
        split.setSizes([700, 580])
        vbox.addWidget(split)
        return page

    # ---- 回调 ----
    def on_done(self, result: dict):
        self.progress.setVisible(False)
        eggs = result.get("eggs", [])
        self.tabs.clear()
        for eg in eggs:
            page = self.build_tab_for_egg(eg)
            self.tabs.addTab(page, f"鸡蛋 {eg['idx']}")
        if self.tabs.count() == 0:
            QMessageBox.warning(self, "提示", "没有可显示的实例结果。")
        self.log("✅ 完成。")

    def on_failed(self, err: str):
        self.progress.setVisible(False)
        self.log(err)
        QMessageBox.critical(self, "出错", err)
        self.stack.setCurrentWidget(self.page_upload)

    # ---- 工具 ----
    def set_pix_to_label(self, lbl: QLabel, qimg: QImage):
        pix = QPixmap.fromImage(qimg)
        scaled = pix.scaled(lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        lbl.setPixmap(scaled)

    def log(self, text: str):
        if hasattr(self, 'log_box') and self.log_box:
            self.log_box.appendPlainText(str(text))

    def clear_results(self):
        if hasattr(self, 'tabs') and self.tabs:
            self.tabs.clear()
        if hasattr(self, 'log_box') and self.log_box:
            self.log_box.clear()


def main():
    app = QApplication(sys.argv)
    w = MainWindow(); w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

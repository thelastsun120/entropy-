from __future__ import annotations

from datetime import datetime

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.complexity_evaluator import evaluate_complexity
from core.entropy_calculator import calculate_global_entropy, calculate_histogram, calculate_local_entropy
from core.image_io import load_image_as_rgb_and_gray
from core.llm_evaluator import evaluate_complexity_with_llm
from core.noise_processor import add_gaussian_noise, add_salt_pepper_noise
from ui.image_panel import ImagePanel
from ui.plot_canvas import MplCanvas


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Entropy Visual Complexity Analyzer")
        self.resize(1380, 860)

        self.original_image = None
        self.gray_image = None
        self.noisy_image = None
        self.local_entropy_map = None
        self.global_entropy = None
        self.noisy_entropy = None

        self.init_ui()
        self.connect_signals()

    def init_ui(self) -> None:
        central = QWidget()
        root_layout = QVBoxLayout(central)

        top_layout = QHBoxLayout()

        self.original_panel = ImagePanel("原始图像")
        self.processed_panel = ImagePanel("灰度 / 加噪图像")
        top_layout.addWidget(self.original_panel, 3)
        top_layout.addWidget(self.processed_panel, 3)

        control_layout = QVBoxLayout()

        self.load_btn = QPushButton("上传图像")
        self.global_btn = QPushButton("计算全局熵")
        self.local_btn = QPushButton("计算局部熵")
        self.noise_btn = QPushButton("添加噪声")
        self.clear_btn = QPushButton("清除结果")
        self.save_btn = QPushButton("保存分析报告")

        self.noise_combo = QComboBox()
        self.noise_combo.addItems(["无噪声", "高斯噪声", "椒盐噪声"])

        self.radius_combo = QComboBox()
        self.radius_combo.addItems(["3", "5", "7", "9", "15"])
        self.radius_combo.setCurrentText("5")

        self.noise_slider = QSlider(Qt.Horizontal)
        self.noise_slider.setRange(0, 100)
        self.noise_slider.setValue(20)
        self.noise_value_label = QLabel("20")

        self.show_hist_cb = QCheckBox("显示灰度直方图")
        self.show_hist_cb.setChecked(True)
        self.show_local_heatmap_cb = QCheckBox("显示局部熵热力图")
        self.show_local_heatmap_cb.setChecked(True)
        self.use_llm_cb = QCheckBox("使用大语言模型评级")
        self.use_llm_cb.setChecked(False)

        control_layout.addWidget(self.load_btn)
        control_layout.addWidget(self.global_btn)
        control_layout.addWidget(self.local_btn)
        control_layout.addWidget(self.noise_btn)
        control_layout.addWidget(self.clear_btn)
        control_layout.addWidget(self.save_btn)

        control_layout.addSpacing(10)
        control_layout.addWidget(QLabel("噪声类型"))
        control_layout.addWidget(self.noise_combo)
        control_layout.addWidget(QLabel("局部熵半径"))
        control_layout.addWidget(self.radius_combo)

        noise_layout = QHBoxLayout()
        noise_layout.addWidget(self.noise_slider)
        noise_layout.addWidget(self.noise_value_label)
        control_layout.addWidget(QLabel("噪声强度"))
        control_layout.addLayout(noise_layout)

        control_layout.addWidget(self.show_hist_cb)
        control_layout.addWidget(self.show_local_heatmap_cb)
        control_layout.addWidget(self.use_llm_cb)
        control_layout.addStretch(1)

        top_layout.addLayout(control_layout, 2)
        root_layout.addLayout(top_layout)

        chart_layout = QHBoxLayout()
        self.hist_canvas = MplCanvas(width=4.5, height=3)
        self.entropy_canvas = MplCanvas(width=4.5, height=3)
        self.comparison_canvas = MplCanvas(width=4.5, height=3)
        chart_layout.addWidget(self.hist_canvas)
        chart_layout.addWidget(self.entropy_canvas)
        chart_layout.addWidget(self.comparison_canvas)
        root_layout.addLayout(chart_layout)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setPlaceholderText("分析结果将在这里显示...")
        root_layout.addWidget(self.result_text, 2)

        self.setCentralWidget(central)

    def connect_signals(self) -> None:
        self.load_btn.clicked.connect(self.load_image)
        self.global_btn.clicked.connect(self.compute_global_entropy)
        self.local_btn.clicked.connect(self.compute_local_entropy)
        self.noise_btn.clicked.connect(self.apply_noise)
        self.clear_btn.clicked.connect(self.clear_results)
        self.save_btn.clicked.connect(self.save_report)
        self.noise_slider.valueChanged.connect(lambda value: self.noise_value_label.setText(str(value)))

    def load_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图像",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)",
        )
        if not file_path:
            return

        try:
            self.original_image, self.gray_image = load_image_as_rgb_and_gray(file_path)
        except ValueError as exc:
            QMessageBox.warning(self, "提示", str(exc))
            return

        self.noisy_image = None
        self.local_entropy_map = None
        self.global_entropy = None
        self.noisy_entropy = None

        self.original_panel.set_image(self.original_image)
        self.processed_panel.set_image(self.gray_image)

        h, w = self.gray_image.shape
        self.result_text.append("图像加载成功，已转换为灰度图。")
        self.result_text.append(f"图像尺寸：{w} × {h}")
        self.result_text.append("")

    def compute_global_entropy(self) -> None:
        if self.gray_image is None:
            QMessageBox.warning(self, "提示", "请先上传图像")
            return

        self.global_entropy = calculate_global_entropy(self.gray_image)
        hist = calculate_histogram(self.gray_image)
        level, description = evaluate_complexity(self.global_entropy)

        if self.show_hist_cb.isChecked():
            self.hist_canvas.plot_histogram(hist)

        self.result_text.append(f"全局熵：{self.global_entropy:.4f} bit")
        self.result_text.append(f"视觉复杂度评级：{level}")
        self.result_text.append(f"分析：{description}")
        self._append_llm_rating()
        self.result_text.append("")

    def compute_local_entropy(self) -> None:
        if self.gray_image is None:
            QMessageBox.warning(self, "提示", "请先上传图像")
            return

        radius = int(self.radius_combo.currentText())
        self.local_btn.setEnabled(False)
        try:
            self.local_entropy_map = calculate_local_entropy(self.gray_image, radius)
        finally:
            self.local_btn.setEnabled(True)

        local_mean = float(np.mean(self.local_entropy_map))
        local_std = float(np.std(self.local_entropy_map))

        if self.show_local_heatmap_cb.isChecked():
            self.entropy_canvas.plot_entropy_map(self.local_entropy_map)

        self.processed_panel.set_image((255 * (self.local_entropy_map / np.max(self.local_entropy_map))).astype(np.uint8))

        self.result_text.append(f"局部熵半径：{radius}")
        self.result_text.append(f"局部熵均值：{local_mean:.4f}")
        self.result_text.append(f"局部熵标准差：{local_std:.4f}")

        if self.global_entropy is not None:
            level, description = evaluate_complexity(self.global_entropy, local_mean, local_std)
            self.result_text.append(f"综合复杂度评级：{level}")
            self.result_text.append(f"分析：{description}")
            self._append_llm_rating(local_mean, local_std)
        self.result_text.append("")

    def apply_noise(self) -> None:
        if self.gray_image is None:
            QMessageBox.warning(self, "提示", "请先上传图像")
            return

        noise_type = self.noise_combo.currentText()
        strength = self.noise_slider.value()

        if noise_type == "高斯噪声":
            sigma = float(strength)
            self.noisy_image = add_gaussian_noise(self.gray_image, sigma=sigma)
        elif noise_type == "椒盐噪声":
            amount = strength / 1000.0
            self.noisy_image = add_salt_pepper_noise(self.gray_image, amount=amount)
        else:
            self.noisy_image = self.gray_image.copy()

        self.noisy_entropy = calculate_global_entropy(self.noisy_image)
        self.processed_panel.set_image(self.noisy_image)

        if self.global_entropy is None:
            self.global_entropy = calculate_global_entropy(self.gray_image)

        delta = self.noisy_entropy - self.global_entropy

        self.comparison_canvas.plot_entropy_comparison(self.global_entropy, self.noisy_entropy)

        self.result_text.append(f"噪声类型：{noise_type}")
        self.result_text.append(f"噪声强度：{strength}")
        self.result_text.append(f"原图熵值：{self.global_entropy:.4f} bit")
        self.result_text.append(f"加噪后熵值：{self.noisy_entropy:.4f} bit")
        self.result_text.append(f"熵值变化：{delta:+.4f} bit")

        if delta > 0:
            self.result_text.append("分析：噪声使灰度分布更加分散，图像不确定性增加。")
        elif delta < 0:
            self.result_text.append("分析：当前噪声处理使灰度分布更集中，熵值下降。")
        else:
            self.result_text.append("分析：噪声前后熵值基本不变。")
        self._append_llm_rating(entropy_delta=delta)
        self.result_text.append("")

    def _append_llm_rating(
        self,
        local_mean: float | None = None,
        local_std: float | None = None,
        entropy_delta: float | None = None,
    ) -> None:
        if not self.use_llm_cb.isChecked():
            return
        if self.global_entropy is None:
            return

        try:
            result = evaluate_complexity_with_llm(
                global_entropy=self.global_entropy,
                local_entropy_mean=local_mean,
                local_entropy_std=local_std,
                noisy_entropy=self.noisy_entropy,
                entropy_delta=entropy_delta,
            )
            self.result_text.append(f"LLM 复杂度评级（{result.model}）：{result.level}")
            self.result_text.append(f"LLM 分析：{result.description}")
        except RuntimeError as exc:
            self.result_text.append(f"LLM 评级不可用：{exc}")

    def clear_results(self) -> None:
        self.original_image = None
        self.gray_image = None
        self.noisy_image = None
        self.local_entropy_map = None
        self.global_entropy = None
        self.noisy_entropy = None

        self.original_panel.clear_panel()
        self.processed_panel.clear_panel()
        self.hist_canvas.clear()
        self.hist_canvas.draw()
        self.entropy_canvas.clear()
        self.entropy_canvas.draw()
        self.comparison_canvas.clear()
        self.comparison_canvas.draw()
        self.result_text.clear()

    def save_report(self) -> None:
        report = self.result_text.toPlainText().strip()
        if not report:
            QMessageBox.information(self, "提示", "当前没有可保存的分析结果")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "保存分析报告",
            f"entropy_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "Text Files (*.txt)",
        )
        if not path:
            return

        with open(path, "w", encoding="utf-8") as f:
            f.write(report)
        QMessageBox.information(self, "提示", "分析报告保存成功")

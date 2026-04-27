# Image Entropy Visual Complexity Analyzer

## 1. 项目简介

这是一个基于 **Python + PyQt5** 的桌面 GUI 工具，用于分析图像灰度分布的信息熵与视觉复杂度。程序支持上传图像、灰度转换、全局熵计算、局部熵热力图、噪声注入分析、灰度直方图和熵值对比图展示。

## 2. 环境依赖

- Python 3.9+
- numpy
- opencv-python
- matplotlib
- PyQt5
- scikit-image

## 3. 安装方式

```bash
pip install -r requirements.txt
```

## 4. 运行方式

```bash
python main.py
```

> 请在 `image_entropy_analyzer/` 目录下执行以上命令。

## 5. 功能说明

1. 上传 JPG / PNG / BMP 图像。
2. 自动展示原图与灰度图。
3. 计算并显示全局信息熵（bit）。
4. 绘制灰度直方图。
5. 输出视觉复杂度评级（低/中/高）。
6. 支持按半径（3/5/7/9/15）计算局部熵。
7. 展示局部熵热力图。
8. 支持添加高斯噪声与椒盐噪声。
9. 显示加噪后熵值与熵值变化，并给出自动分析结论。
10. 支持导出文本分析报告。

## 6. 信息熵公式说明

全局灰度熵定义如下：

\[
H = -\sum_{i=0}^{255} p_i\log_2 p_i
\]

- \(H\)：图像信息熵（单位 bit）
- \(p_i\)：灰度值 \(i\) 的概率
- 灰度级范围：0 到 255

熵越高通常表示灰度分布越分散、不确定性越大。

## 7. 图像复杂度评级规则

默认根据全局熵评级：

- `H < 4.0`：低复杂度
- `4.0 <= H < 6.5`：中等复杂度
- `H >= 6.5`：高复杂度

当局部熵可用时，使用综合评分：

\[
score = 0.7 \times global\_entropy + 0.3 \times local\_entropy\_mean
\]

再按同样阈值进行分级。

## 8. 噪声影响分析说明

程序支持：

- 高斯噪声（通过 `sigma` 控制强度）
- 椒盐噪声（通过 `amount` 控制比例）

分析输出：

- 原图熵值
- 加噪后熵值
- 熵值变化量
- 自动解释（熵值上升通常对应不确定性增强，未必意味着有效信息增加）

## 9. 项目截图占位说明

可在此处补充课程展示截图：

- 主界面截图（原图 + 灰度图 + 控制区）
- 局部熵热力图截图
- 噪声前后熵值对比截图

---

## 目录结构

```text
image_entropy_analyzer/
├── main.py
├── requirements.txt
├── README.md
├── core/
│   ├── __init__.py
│   ├── image_io.py
│   ├── entropy_calculator.py
│   ├── noise_processor.py
│   └── complexity_evaluator.py
├── ui/
│   ├── __init__.py
│   ├── main_window.py
│   ├── image_panel.py
│   └── plot_canvas.py
└── assets/
    └── sample_images/
```

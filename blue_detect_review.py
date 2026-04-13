"""
变更审查文件（用于在右侧编辑器查看）：

每个变更块用标签 `CHANGE_<id>` 标记，并在块下方显示建议的替换代码。
在你决定后，编辑 `decisions.json`（将对应 id 设为 true/false），然后运行 `apply_decisions.py`。

打开这个文件在右侧查看并决定每个变更是否“保留”或“放弃”。
"""

# 原始文件摘录（简化以便对比）
ORIGINAL = '''
import cv2
import numpy as np

def detect_blue_cell_center(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 蓝色提取区间可调
    lower_blue = np.array([100, 80, 80])
    upper_blue = np.array([130, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 形态学处理去噪
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    if num_labels <= 1:
        raise ValueError("未检测到蓝色格子")

    # # 找面积最大（排除背景索引0）
    areas = stats[1:, cv2.CC_STAT_AREA]
    max_idx = np.argmax(areas) + 1

    cx, cy = centroids[max_idx]
    return float(cx), float(cy)
'''


# ---------------------------
# CHANGE_YELLOW_RANGE_1: 扩大黄色提取“空间范围”（HSV 阈值范围 + 形态学膨胀）
# 说明:
# - 你现在的 `blue_detect.py` 里只有蓝色检测；这里用一个自洽变更块，直接替换整个 `detect_blue_cell_center`
# - 黄色范围扩大：H 从常见 15-35 扩大到 10-50；S/V 下限放宽到 50，能覆盖更浅的黄
# - 空间范围扩大：对 yellow_mask 做更强的闭运算 + 膨胀（避免黄区域断裂、增强连通域）

CHANGE_YELLOW_RANGE_1_OLD = '''
def detect_blue_cell_center(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 蓝色提取区间可调
    lower_blue = np.array([100, 80, 80])
    upper_blue = np.array([130, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 形态学处理去噪
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    if num_labels <= 1:
        raise ValueError("未检测到蓝色格子")

    # # 找面积最大（排除背景索引0）
    areas = stats[1:, cv2.CC_STAT_AREA]
    max_idx = np.argmax(areas) + 1

    cx, cy = centroids[max_idx]
    return float(cx), float(cy)
'''

CHANGE_YELLOW_RANGE_1_NEW = '''
def detect_blue_cell_center(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 蓝色提取区间可调
    lower_blue = np.array([100, 80, 80])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 新增：黄色提取区间（扩大范围）
    # - H: 10~50 覆盖偏橙黄/偏绿黄
    # - S/V: 下限放宽到 50 覆盖更浅的黄
    lower_yellow = np.array([10, 50, 50])
    upper_yellow = np.array([50, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # 形态学处理去噪/扩张（黄色“空间范围”更大）
    k3 = np.ones((3, 3), np.uint8)
    k5 = np.ones((5, 5), np.uint8)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, k3)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, k5)
    yellow_mask = cv2.dilate(yellow_mask, k5, iterations=1)

    def _largest_centroid(mask):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        if num_labels <= 1:
            return None
        areas = stats[1:, cv2.CC_STAT_AREA]
        max_idx = np.argmax(areas) + 1
        cx, cy = centroids[max_idx]
        return float(cx), float(cy)

    # 兼容原逻辑：优先返回蓝色；蓝色缺失时再用黄色兜底
    blue_center = _largest_centroid(blue_mask)
    if blue_center is not None:
        return blue_center

    yellow_center = _largest_centroid(yellow_mask)
    if yellow_center is not None:
        return yellow_center

    raise ValueError("未检测到蓝色或黄色格子")
'''


def render_review():
    print("=== 原始 detect_blue_cell_center ===\n")
    print(ORIGINAL)
    print("\n=== 建议变更（按 ID 对应 decisions.json 填写保留/放弃） ===\n")
    print("[CHANGE_YELLOW_RANGE_1] 扩大黄色提取范围（HSV + 空间膨胀）:\n", CHANGE_YELLOW_RANGE_1_NEW)


if __name__ == '__main__':
    render_review()

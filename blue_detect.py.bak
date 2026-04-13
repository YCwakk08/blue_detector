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




def locate_grid_cell(img, cx, cy):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 检测黑色网格线
    _, grid_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    horizontal_proj = np.sum(grid_mask > 0, axis=1)
    vertical_proj = np.sum(grid_mask > 0, axis=0)

    # 聚类线段位置函数
    def cluster_lines(lines, gap=5):
        if len(lines) == 0:
            return []
        clusters = [[lines[0]]]
        for l in lines[1:]:
            if l - clusters[-1][-1] <= gap:
                clusters[-1].append(l)
            else:
                clusters.append([l])
        return [int(np.mean(c)) for c in clusters]

    # 选线
    h_indices = np.where(horizontal_proj > w * 0.5)[0]
    v_indices = np.where(vertical_proj > h * 0.5)[0]

    h_lines = cluster_lines(h_indices)
    v_lines = cluster_lines(v_indices)

    if len(h_lines) < 2 or len(v_lines) < 2:
        raise ValueError("未成功检测到网格线")

    # 根据中心位置确定行列
    row = np.searchsorted(h_lines, cy) - 1
    col = np.searchsorted(v_lines, cx) - 1

    return int(row), int(col)


def process(img):
    cx, cy = detect_blue_cell_center(img)
    row, col = locate_grid_cell(img, cx, cy)
    return cx, cy, row, col


# ===============================
# Example Usage
# ===============================
if __name__ == "__main__":
    img = cv2.imread("screen_shoot.png")
    cx, cy, row, col = process(img)
    print("像素坐标:", (cx, cy))
    print("网格坐标:", (row, col))

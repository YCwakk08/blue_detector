import cv2
import numpy as np
from mss import mss
from mss.tools import to_png  # Optional, if needed for saving
import time

def detect_blue_cell_center(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 蓝色提取区间可调
    lower_blue = np.array([100, 80, 80])
    upper_blue = np.array([130, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 形态学处理去噪
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask)

    if num_labels <= 1:
        raise ValueError("未检测到蓝色格子")

    # 找面积最大（排除背景索引0）
    areas = stats[1:, cv2.CC_STAT_AREA]
    max_idx = np.argmax(areas) + 1

    cx, cy = centroids[max_idx]
    return float(cx), float(cy)



# 主函数：实时屏幕捕获和检测
def realtime_screen_detection():
    lower_blue = np.array([100, 80, 80])
    upper_blue = np.array([130, 255, 255])
    with mss() as sct:
        # 定义捕获区域：全屏示例，根据需要调整（top, left, width, height）
        monitor = sct.monitors[1]  # monitors[0] 是所有屏幕，monitors[1] 通常是主屏

        while True:
            # 捕获屏幕
            screenshot = sct.grab(monitor)
            
            img = np.array(screenshot)  # 转换为numpy数组 (BGRA)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # 转换为BGR

            try:
                start_time = time.time()
                cx, cy = detect_blue_cell_center(img)
                end_time = time.time()
                print(f"蓝色网格中心坐标: ({cx}, {cy}), 耗时: {end_time - start_time:.2f}秒")
                # 这里可以返回坐标，如果在其他地方调用
                # return cx, cy  # 但由于是循环，视情况使用
            except ValueError as e:
                print(e)  # 未检测到时打印错误

            # 可选：显示捕获的图像和mask（调试用）
            cv2.imshow('Screen Capture', img)
            mask = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), lower_blue, upper_blue)
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            cv2.imshow('Blue Mask', mask)

            # 按'q'退出循环
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

# 运行
if __name__ == "__main__":
    realtime_screen_detection()
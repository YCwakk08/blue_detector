import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / 'blue_detect.py'
DECISIONS = ROOT / 'decisions.json'

CHANGES = {
    'CHANGE_YELLOW_RANGE_1': {
        'old': (
            "def detect_blue_cell_center(img):\n"
            "    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)\n"
            "\n"
            "    # 蓝色提取区间可调\n"
            "    lower_blue = np.array([100, 80, 80])\n"
            "    upper_blue = np.array([130, 255, 255])\n"
            "\n"
            "    mask = cv2.inRange(hsv, lower_blue, upper_blue)\n"
            "\n"
            "    # 形态学处理去噪\n"
            "    kernel = np.ones((3, 3), np.uint8)\n"
            "    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)\n"
            "\n"
            "    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)\n"
            "\n"
            "    if num_labels <= 1:\n"
            "        raise ValueError(\"未检测到蓝色格子\")\n"
            "\n"
            "    # # 找面积最大（排除背景索引0）\n"
            "    areas = stats[1:, cv2.CC_STAT_AREA]\n"
            "    max_idx = np.argmax(areas) + 1\n"
            "\n"
            "    cx, cy = centroids[max_idx]\n"
            "    return float(cx), float(cy)\n"
        ),
        'new': (
            "def detect_blue_cell_center(img):\n"
            "    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)\n"
            "\n"
            "    # 蓝色提取区间可调\n"
            "    lower_blue = np.array([100, 80, 80])\n"
            "    upper_blue = np.array([130, 255, 255])\n"
            "    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)\n"
            "\n"
            "    # 新增：黄色提取区间（扩大范围）\n"
            "    # - H: 10~50 覆盖偏橙黄/偏绿黄\n"
            "    # - S/V: 下限放宽到 50 覆盖更浅的黄\n"
            "    lower_yellow = np.array([10, 50, 50])\n"
            "    upper_yellow = np.array([50, 255, 255])\n"
            "    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)\n"
            "\n"
            "    # 形态学处理去噪/扩张（黄色“空间范围”更大）\n"
            "    k3 = np.ones((3, 3), np.uint8)\n"
            "    k5 = np.ones((5, 5), np.uint8)\n"
            "    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, k3)\n"
            "    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, k5)\n"
            "    yellow_mask = cv2.dilate(yellow_mask, k5, iterations=1)\n"
            "\n"
            "    def _largest_centroid(mask):\n"
            "        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)\n"
            "        if num_labels <= 1:\n"
            "            return None\n"
            "        areas = stats[1:, cv2.CC_STAT_AREA]\n"
            "        max_idx = np.argmax(areas) + 1\n"
            "        cx, cy = centroids[max_idx]\n"
            "        return float(cx), float(cy)\n"
            "\n"
            "    # 兼容原逻辑：优先返回蓝色；蓝色缺失时再用黄色兜底\n"
            "    blue_center = _largest_centroid(blue_mask)\n"
            "    if blue_center is not None:\n"
            "        return blue_center\n"
            "\n"
            "    yellow_center = _largest_centroid(yellow_mask)\n"
            "    if yellow_center is not None:\n"
            "        return yellow_center\n"
            "\n"
            "    raise ValueError(\"未检测到蓝色或黄色格子\")\n"
        ),
    },
}


def apply_change_text(content, change_id):
    c = CHANGES[change_id]
    if 'insert' in c:
        # insert after the marker
        marker = c['where_after']
        if marker not in content:
            raise RuntimeError(f"未找到插入点 for {change_id}")
        return content.replace(marker, marker + c['insert'])
    else:
        old = c['old']
        new = c['new']
        if old not in content:
            raise RuntimeError(f"原始片段未找到，无法替换 {change_id}")
        return content.replace(old, new)


def main():
    if not DECISIONS.exists():
        print("找不到 decisions.json，请先在工作区编辑并填写保留(true)/放弃(false)")
        sys.exit(1)

    decisions = json.loads(DECISIONS.read_text())

    content = TARGET.read_text()

    for cid, val in decisions.items():
        if cid not in CHANGES:
            print(f"忽略未知变更 {cid}（当前 apply_decisions.py 未定义）")
            continue
        if val is True:
            print(f"应用变更 {cid} ...")
            try:
                content = apply_change_text(content, cid)
            except Exception as e:
                print(f"应用 {cid} 失败: {e}")
                sys.exit(2)
        elif val is False:
            print(f"跳过变更 {cid}（放弃）")
        else:
            print(f"未决策 {cid}，跳过")

    # 备份原文件
    bak = TARGET.with_suffix('.py.bak')
    if not bak.exists():
        bak.write_text(TARGET.read_text())
        print(f"已创建备份: {bak.name}")

    TARGET.write_text(content)
    print(f"已将选定变更应用到 {TARGET.name}")


if __name__ == '__main__':
    main()

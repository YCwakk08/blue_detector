import cv2
import numpy as np

# --- 读取图像 ---
img = cv2.imread("process/screen_shoot2.png")
h, w, _ = img.shape

# --- 转HSV用于蓝色提取 ---
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imwrite("process/01_hsv.png", hsv)  # 保存HSV效果图（H通道）
# lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
# cv2.imwrite("process/01_lab.png", lab)  # 保存Lab效果图（L通道）
print(f"✓ 原始图像大小: {w}x{h}")

# 蓝色区间 (可微调)
lower_blue = np.array([100, 80, 80])
upper_blue = np.array([130, 255, 255])

mask = cv2.inRange(hsv, lower_blue, upper_blue)
cv2.imwrite("process/02_mask_blue.png", mask)  # 保存蓝色掩码

# --- 形态学处理去噪 ---
kernel = np.ones((3,3), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # 开运算：腐蚀→膨胀，去除周边噪声
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 闭运算：膨胀→腐蚀，填充内部孔洞
cv2.imwrite("process/03_mask_closed.png", mask)  # 保存形态学处理后的掩码

# 如果需要开运算：
# mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # 开运算：腐蚀→膨胀，去除周边噪声

# --- 连通域分析找最大蓝色区域 ---
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
# num_labels: 连通域总数（包括背景标签0）
#   = 所有独立的像素块数 + 1（背景）
#   例如：有3个独立的蓝色方块 → num_labels = 4（标签0、1、2、3）
# 
# labels: 标签图，shape=(h,w)，像素值=所属连通域的标签
#   例如：labels[50, 60] = 2，表示第50行60列属于标签2的连通域
# 
# stats: shape=(num_labels, 5)，每行=[x,y,width,height,area]
#   stats[0] = 背景的统计（矩形包围所有黑色像素）
#   stats[1] = 标签1连通域的统计
#   stats[2] = 标签2连通域的统计
#   ...
#   排序规律：按像素扫描顺序（上→下，左→右），先找到的连通域标签越小
# 
# centroids: shape=(num_labels, 2)，每行=[cx, cy]，中心坐标

print(f"✓ 连通域总数：{num_labels}（包括背景）")
print(f"  标签范围：0（背景）到 {num_labels-1}（最后的连通域）")

# 找出面积最大（排除背景0）
areas = stats[1:, cv2.CC_STAT_AREA]  # 跳过标签0（背景），取所有连通域的面积
max_idx = np.argmax(areas) + 1  # +1是因为跳过了标签0
print(f"✓ 最大连通域标签：{max_idx}，面积：{areas[max_idx-1]} 像素")

# 蓝色格子的坐标信息
x, y, bw, bh, area = stats[max_idx]
cx, cy = centroids[max_idx]   # 图像像素中心坐标

print("蓝格子像素中心坐标:", (cx, cy))

# --- 检测网格线 (检测黑线) ---
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 网格线为黑，通过阈值提取线条
_, grid_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
cv2.imwrite("process/04_grid_mask.png", grid_mask)  # 保存网格线掩码

# 水平投影找到水平线位置
horizontal_proj = np.sum(grid_mask > 0, axis=1)  # 统计每行的线条像素数
h_lines = np.where(horizontal_proj > w * 0.5)[0]  # 筛选：像素数 > 宽度50%（完整的线）

# 垂直投影找到垂直线位置
vertical_proj = np.sum(grid_mask > 0, axis=0)  # 统计每列的线条像素数
v_lines = np.where(vertical_proj > h * 0.5)[0]  # 筛选：像素数 > 高度50%（完整的线）

# 对线位置聚类合并（避免线宽造成重复）
# 原因：一条线宽度为3像素，会产生3个相邻的行/列号，需要合并成1个
def cluster_lines(lines, gap=5):
    """
    聚类相邻的行/列号，把同一条线的多个像素位置合并成一个
    
    参数：
        lines: ndarray，排序后的行号或列号数组，如 [10, 11, 12, 30, 31, 32, 50]
        gap: int，相邻判断的间距阈值（默认5像素）
    
    返回：
        list，聚类后的位置列表，如 [11, 31, 50]
    
    实现细节演示：
    -------
    例如输入：lines = [10, 11, 12, 30, 31, 32, 50]，gap = 5
    
    第1步：初始化
        clusters = [[10]]  # 第一个元素自成一簇
        
    第2步：遍历剩余元素 [11, 12, 30, 31, 32, 50]
        
        处理11：
            l=11, 上一簇末尾=10
            11 - 10 = 1 ≤ 5？是 ✓
            → 加入上一簇：clusters = [[10, 11]]
        
        处理12：
            l=12, 上一簇末尾=11
            12 - 11 = 1 ≤ 5？是 ✓
            → 加入上一簇：clusters = [[10, 11, 12]]
        
        处理30：
            l=30, 上一簇末尾=12
            30 - 12 = 18 ≤ 5？否 ✗
            → 新建簇：clusters = [[10, 11, 12], [30]]
        
        处理31：
            l=31, 上一簇末尾=30
            31 - 30 = 1 ≤ 5？是 ✓
            → 加入上一簇：clusters = [[10, 11, 12], [30, 31]]
        
        处理32：
            l=32, 上一簇末尾=31
            32 - 31 = 1 ≤ 5？是 ✓
            → 加入上一簇：clusters = [[10, 11, 12], [30, 31, 32]]
        
        处理50：
            l=50, 上一簇末尾=32
            50 - 32 = 18 ≤ 5？否 ✗
            → 新建簇：clusters = [[10, 11, 12], [30, 31, 32], [50]]
    
    第3步：求均值并取整
        [10, 11, 12] → mean = 11
        [30, 31, 32] → mean = 31
        [50] → mean = 50
        返回 [11, 31, 50]
    """
    if len(lines) == 0:
        return []
    
    clusters = [[lines[0]]]  # 第一个元素作为第一个簇的起点
    
    for l in lines[1:]:  # 遍历后续所有元素
        if l - clusters[-1][-1] <= gap:  # 与上一簇的末尾元素比较距离
            clusters[-1].append(l)  # 加入同一簇
        else:
            clusters.append([l])  # 开启新簇
    
    # 对每个簇求平均值，取整后返回
    return [int(np.mean(c)) for c in clusters]

h_lines = cluster_lines(h_lines)
v_lines = cluster_lines(v_lines)
print(f"✓ 聚类后水平线数：{len(h_lines)}，位置：{h_lines[:5]}..." if len(h_lines) > 5 else f"✓ 聚类后水平线数：{len(h_lines)}，位置：{h_lines}")
print(f"✓ 聚类后垂直线数：{len(v_lines)}，位置：{v_lines[:5]}..." if len(v_lines) > 5 else f"✓ 聚类后垂直线数：{len(v_lines)}，位置：{v_lines}")

# --- 利用行列线判断所在网格 ---
row = np.searchsorted(h_lines, cy) - 1
col = np.searchsorted(v_lines, cx) - 1

print("蓝格子网格坐标 (row, col):", (row, col))

import torch
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import numpy as np

# 加载模型
print("正在加载SAM模型...")
sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
device = "cuda" if torch.cuda.is_available() else "cpu"
sam.to(device)
print(f"使用设备: {device}")

# 创建掩码生成器
mask_generator = SamAutomaticMaskGenerator(sam)

# 创建一张测试图（彩色方块）
print("创建测试图片...")
image = np.zeros((512, 512, 3), dtype=np.uint8)
image[:256, :256] = [255, 0, 0]      # 红色
image[:256, 256:] = [0, 255, 0]      # 绿色
image[256:, :256] = [0, 0, 255]      # 蓝色
image[256:, 256:] = [255, 255, 0]    # 黄色

# 生成掩码
print("生成分割掩码...")
masks = mask_generator.generate(image)
print(f"检测到 {len(masks)} 个对象")

# 可视化结果
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("原始图片")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(image)
for mask in masks:
    color = np.random.rand(3)
    plt.contour(mask['segmentation'], colors=[color], linewidths=2)
plt.title(f"SAM分割结果 ({len(masks)}个对象)")
plt.axis('off')

plt.tight_layout()
plt.savefig("sam_result.png", dpi=150, bbox_inches='tight')
print("结果已保存到 sam_result.png")

plt.show()
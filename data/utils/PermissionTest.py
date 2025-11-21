# simple_test.py - 最简单的权限测试
import cv2
import numpy as np
import os

# 创建一个 50x50 的全黑图片
black_image = np.zeros((50, 50, 3), np.uint8)

# 尝试保存到当前目录
try:
    success = cv2.imwrite('test_permission.jpg', black_image)
    if success:
        print("✅ 成功保存测试图片！")
        print(f"📁 文件位置: {os.path.abspath('test_permission.jpg')}")
        
        # 检查文件信息
        if os.path.exists('test_permission.jpg'):
            size = os.path.getsize('test_permission.jpg')
            print(f"📊 文件大小: {size} 字节")
            
            # # 清理
            # os.remove('test_permission.jpg')
            # print("🧹 已清理测试文件")
        else:
            print("❌ 文件保存成功但不存在")
    else:
        print("❌ 保存失败 - 可能是权限问题")
        
except Exception as e:
    print(f"💥 错误: {e}")
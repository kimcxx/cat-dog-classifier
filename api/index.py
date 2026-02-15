"""
Vercel部署入口点
将Flask应用适配到Vercel的无服务器环境
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入Flask应用
from web.app import app

# Vercel需要这个变量
handler = app
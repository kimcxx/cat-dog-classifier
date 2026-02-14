"""
健壮版的猫狗分类 Web 服务器
添加更好的错误处理和日志记录
支持.zip格式的模型文件
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import os
import numpy as np
import traceback
import logging
import zipfile
import tempfile

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static')
CORS(app)

# 支持的模型版本
MODEL_VERSIONS = {
    'improved': {
        'path': 'cat_dog_model.zip',  # 相对于根目录
        'arch': 'resnet50',
        'description': '改进模型（ResNet50）'
    }
}

# 全局变量
current_model = None
device = None
transform = None
CLASS_NAMES = ['猫', '狗']
CONFIDENCE_THRESHOLD = 0.85  # 提高置信度阈值
MIN_PROB_DIFF = 0.30  # 最小概率差异（30%）


def load_model(version='improved'):
    """加载指定版本的模型"""
    global current_model, device, transform
    
    if version not in MODEL_VERSIONS:
        logger.error(f"不支持的模型版本: {version}")
        version = 'improved'
    
    model_info = MODEL_VERSIONS[version]
    model_path = model_info['path']
    
    # 获取绝对路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)
    absolute_path = os.path.join(parent_dir, model_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    logger.info(f"加载模型: {model_info['description']}")
    logger.info(f"模型路径: {absolute_path}")
    
    # 检查模型文件是否存在
    if not os.path.exists(absolute_path):
        logger.error(f"模型文件不存在: {absolute_path}")
        return False
    
    try:
        # 如果是.zip文件，先解压
        if model_path.endswith('.zip'):
            logger.info("检测到.zip文件，正在解压...")
            with zipfile.ZipFile(absolute_path, 'r') as zip_ref:
                # 创建临时目录
                temp_dir = tempfile.mkdtemp()
                zip_ref.extractall(temp_dir)
                
                # 查找.pth文件
                pth_files = [f for f in os.listdir(temp_dir) if f.endswith('.pth')]
                if not pth_files:
                    logger.error("在.zip文件中未找到.pth模型文件")
                    return False
                
                pth_file = os.path.join(temp_dir, pth_files[0])
                logger.info(f"找到模型文件: {pth_file}")
                checkpoint = torch.load(pth_file, map_location=device)
        else:
            # 直接加载.pth文件
            checkpoint = torch.load(absolute_path, map_location=device)
        
        # 根据架构创建模型
        if model_info['arch'] == 'resnet50':
            model = models.resnet50(pretrained=False)
            num_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 2)
            )
        else:
            logger.error(f"不支持的模型架构: {model_info['arch']}")
            return False
        
        # 加载权重
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        model = model.to(device)
        current_model = model
        
        # 图像预处理
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        logger.info(f"模型加载完成: {model_info['description']}")
        if 'val_acc' in checkpoint:
            logger.info(f"模型验证准确率: {checkpoint['val_acc']:.2f}%")
        
        return True
        
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        logger.error(traceback.format_exc())
        return False


def predict_image(image):
    """预测单张图像"""
    if current_model is None:
        if not load_model():
            return {
                'error': '模型未加载',
                'success': False
            }
    
    try:
        # 基础预测
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = current_model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            probs = probabilities[0].cpu().numpy()
        
        predicted_class = CLASS_NAMES[predicted.item()]
        confidence_score = confidence.item() * 100
        
        # 获取两个类别的概率
        cat_prob = probs[0] * 100
        dog_prob = probs[1] * 100
        max_prob = max(cat_prob, dog_prob)
        prob_diff = abs(cat_prob - dog_prob)
        
        # 改进的拒绝逻辑
        is_uncertain = False
        reason = ''
        suggestion = ''
        
        # 情况1: 置信度过低
        if max_prob < CONFIDENCE_THRESHOLD * 100:
            is_uncertain = True
            reason = 'low_confidence'
            suggestion = '模型对预测结果不够自信，这可能不是猫或狗'
            predicted_class = '既不是猫也不是狗'
        
        # 情况2: 两个类别概率太接近
        elif prob_diff < MIN_PROB_DIFF * 100:
            is_uncertain = True
            reason = 'ambiguous'
            suggestion = '图片特征不明确，可能既像猫又像狗，或者不是猫狗'
            predicted_class = '难以确定'
        
        # 情况3: 中等置信度但概率差异不大（针对人物图片等）
        elif max_prob < 90 and prob_diff < 50:
            is_uncertain = True
            reason = 'medium_confidence'
            suggestion = '这可能不是猫或狗，请上传更清晰的猫狗图片'
            predicted_class = '可能不是猫或狗'
        
        if is_uncertain:
            return {
                'predicted': predicted_class,
                'confidence': round(float(max_prob), 2),
                'probabilities': {
                    '猫': round(float(cat_prob), 2),
                    '狗': round(float(dog_prob), 2)
                },
                'is_uncertain': True,
                'reason': reason,
                'suggestion': suggestion,
                'model_version': '改进模型（ResNet50）'
            }
        else:
            return {
                'predicted': predicted_class,
                'confidence': round(float(confidence_score), 2),
                'probabilities': {
                    '猫': round(float(cat_prob), 2),
                    '狗': round(float(dog_prob), 2)
                },
                'is_uncertain': False,
                'model_version': '改进模型（ResNet50）'
            }
            
    except Exception as e:
        logger.error(f"预测失败: {e}")
        logger.error(traceback.format_exc())
        return {
            'error': f'预测失败: {str(e)}',
            'success': False
        }


@app.route('/')
def index():
    """返回主页"""
    return send_from_directory('static', 'index.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查"""
    try:
        model_loaded = current_model is not None
        return jsonify({
            'status': 'healthy' if model_loaded else 'unhealthy',
            'model_loaded': model_loaded,
            'model': '改进模型（ResNet50）' if model_loaded else None,
            'device': str(device) if device else None
        })
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    """处理图片预测请求"""
    try:
        logger.info("收到预测请求")
        
        # 检查是否有文件上传
        if 'image' not in request.files:
            logger.warning("未找到上传的图片字段")
            return jsonify({'error': '未找到上传的图片'}), 400
        
        file = request.files['image']
        logger.info(f"文件名: {file.filename}")
        
        # 检查文件名是否为空
        if file.filename == '':
            logger.warning("文件名为空")
            return jsonify({'error': '未选择文件'}), 400
        
        # 读取图片
        image_bytes = file.read()
        logger.info(f"图片大小: {len(image_bytes)} bytes")
        
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        logger.info(f"图像尺寸: {image.size}")
        
        # 预测
        logger.info("开始预测...")
        result = predict_image(image)
        logger.info(f"预测结果: {result}")
        
        if 'error' in result:
            return jsonify({'success': False, 'error': result['error']}), 500
        
        return jsonify({'success': True, 'result': result})
        
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        logger.error(f"预测失败: {error_msg}")
        logger.error(f"错误追踪: {error_trace}")
        return jsonify({'error': f'预测失败: {error_msg}'}), 500


@app.route('/<path:filename>')
def static_files(filename):
    """提供静态文件"""
    return send_from_directory('static', filename)


if __name__ == '__main__':
    # 启动前加载默认模型
    logger.info("正在加载模型...")
    if not load_model('improved'):
        logger.error("模型加载失败，服务器可能无法正常工作")
    
    logger.info("\n启动服务器...")
    logger.info("访问地址: http://localhost:5000")
    logger.info("API 端点:")
    logger.info("  GET  /api/health      - 健康检查")
    logger.info("  POST /predict         - 预测图片")
    logger.info("\n拒绝机制配置:")
    logger.info(f"  置信度阈值: {CONFIDENCE_THRESHOLD*100}%")
    logger.info(f"  最小概率差异: {MIN_PROB_DIFF*100}%")
    
    app.run(host='0.0.0.0', port=5000, debug=False)  # 关闭debug模式以获得更好的稳定性

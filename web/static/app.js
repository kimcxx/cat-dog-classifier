// DOM 元素
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const previewArea = document.getElementById('previewArea');
const previewImage = document.getElementById('previewImage');
const resultArea = document.getElementById('resultArea');
const loadingArea = document.getElementById('loadingArea');
const resultEmoji = document.getElementById('resultEmoji');
const resultLabel = document.getElementById('resultLabel');
const confidenceValue = document.getElementById('confidenceValue');
const confidenceBar = document.getElementById('confidenceBar');
const catProb = document.getElementById('catProb');
const dogProb = document.getElementById('dogProb');
const suggestionText = document.getElementById('suggestionText');

// 拖放事件
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

// 文件选择
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

// 处理文件
function handleFile(file) {
    // 检查是否是图片
    if (!file.type.startsWith('image/')) {
        alert('请选择图片文件！');
        return;
    }

    // 显示预览
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        dropZone.style.display = 'none';
        previewArea.style.display = 'block';
        resultArea.style.display = 'none';
        suggestionText.style.display = 'none';

        // 发送到服务器进行识别
        predictImage(file);
    };
    reader.readAsDataURL(file);
}

// 预测图片
async function predictImage(file) {
    const formData = new FormData();
    formData.append('image', file);

    // 显示加载中
    loadingArea.style.display = 'block';

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        loadingArea.style.display = 'none';

        if (data.success) {
            displayResult(data.result);
        } else {
            alert('预测失败: ' + (data.error || '未知错误'));
        }
    } catch (error) {
        loadingArea.style.display = 'none';
        alert('请求失败: ' + error.message);
    }
}

// 显示结果
function displayResult(result) {
    const isCat = result.predicted === '猫';
    const isDog = result.predicted === '狗';
    const isUncertain = result.is_uncertain;
    
    // 确保概率对象存在
    const probabilities = result.probabilities || {};
    const catProbValue = probabilities['猫'] || 0;
    const dogProbValue = probabilities['狗'] || 0;

    // 设置主要结果
    if (isUncertain) {
        // 不确定的情况
        resultEmoji.textContent = '❓';
        resultLabel.textContent = result.predicted;
        confidenceValue.textContent = result.confidence;

        // 设置进度条为灰色
        confidenceBar.style.background = '#ccc';
        confidenceBar.style.width = result.confidence + '%';

        // 设置概率为灰色
        catProb.style.color = '#999';
        dogProb.style.color = '#999';
        catProb.textContent = catProbValue + '%';
        dogProb.textContent = dogProbValue + '%';

        // 显示建议
        if (result.suggestion) {
            suggestionText.textContent = result.suggestion;
            suggestionText.style.display = 'block';
        }
    } else {
        // 确定的情况
        resultEmoji.textContent = isCat ? '🐱' : '🐶';
        resultLabel.textContent = result.predicted;
        confidenceValue.textContent = result.confidence;

        // 设置进度条
        confidenceBar.style.background = 'linear-gradient(90deg, #667eea 0%, #764ba2 100%)';
        confidenceBar.style.width = result.confidence + '%';

        // 设置概率
        catProb.style.color = isCat ? '#667eea' : '#999';
        dogProb.style.color = isDog ? '#667eea' : '#999';
        catProb.textContent = catProbValue + '%';
        dogProb.textContent = dogProbValue + '%';
        
        // 隐藏建议
        suggestionText.style.display = 'none';
    }

    // 显示结果区域
    resultArea.style.display = 'block';
}

// 移除图片
function removeImage() {
    dropZone.style.display = 'block';
    previewArea.style.display = 'none';
    resultArea.style.display = 'none';
    suggestionText.style.display = 'none';
    resultEmoji.textContent = '';
    resultLabel.textContent = '';
    confidenceValue.textContent = '0';
    catProb.textContent = '0%';
    dogProb.textContent = '0%';
    fileInput.value = '';
}

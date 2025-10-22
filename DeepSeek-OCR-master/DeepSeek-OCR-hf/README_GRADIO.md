# DeepSeek OCR Gradio 界面使用说明

这是一个基于Gradio的Web界面，用于DeepSeek OCR模型的文档识别，具有实时流式输出、Markdown渲染和图片提取功能。

## 🎉 功能特点

### 📁 文件支持
- **图片格式**: JPG, PNG, BMP, TIFF, WEBP
- **PDF文档**: 自动转换为图片进行逐页识别，支持多页文档批量处理

### 🎯 OCR模式
- **Free OCR**: 纯文本识别模式
- **Convert to Markdown**: 转换为Markdown格式（推荐），支持结构化识别

### ⚡ 核心功能

#### 1. 实时流式输出
- 识别过程中实时显示结果，无需等待完成
- 类似打字机效果，逐字逐句显示
- 大幅提升用户体验

#### 2. Markdown实时渲染
- 自动渲染标题、列表、表格等格式
- 支持公式、代码块等复杂元素
- 图片自动嵌入显示（使用base64编码）

#### 3. 智能图片提取
- 自动识别文档中的图片区域
- 根据坐标精确裁切图片
- 保存为高质量JPEG格式（quality=95）

#### 4. 多标签页展示
- **📝 Markdown渲染**: 格式化显示，图片嵌入，最佳阅读体验
- **📄 纯文本**: 清理后的纯文本，支持一键复制
- **🔧 原始输出**: 包含完整标注信息，可用于调试和二次开发
- **🖼️ 提取的图片**: 画廊展示所有提取的图片，支持单独下载

#### 5. 高级设置
- **Base Size**: 基础尺寸，影响识别精度（512-1280）
- **Image Size**: 图像尺寸（512-1280）
- **Crop Mode**: 裁剪模式开关
- 内置多种预设配置

#### 6. 其他特性
- 停止按钮：可随时中断识别任务
- 示例文件：快速体验功能
- 持久化输出：识别结果保存在 `gradio_outputs/` 目录
- 调试信息自动清理：输出结果更整洁

## 🚀 安装依赖

### 基础依赖
```bash
pip install gradio pymupdf
```

### 完整依赖列表
```bash
pip install transformers torch pillow gradio pymupdf
```

确保已安装CUDA和Flash Attention 2（用于加速推理）：
```bash
pip install flash-attn --no-build-isolation
```

## 📖 使用方法

### 1. 启动界面

```bash
cd /path/to/DeepSeek-OCR-hf
python gradio_ocr.py
```

启动后会显示：
```
Loading model...
Model loaded successfully!
Running on local URL:  http://0.0.0.0:7860
```

### 2. 访问界面

**本地访问**:
```
http://localhost:7860
```

**远程访问**:
```
http://服务器IP:7860
```

如果启用了 `share=True`，还会生成一个公网链接（72小时有效）。

### 3. 使用步骤

#### 基础使用流程

1. **上传文件**
   - 点击"📁 上传图片或PDF文件"
   - 选择要识别的文件
   - 支持拖拽上传

2. **选择OCR模式**
   - `Free OCR`: 快速纯文本识别
   - `Convert to Markdown`: 结构化识别（推荐）

3. **调整参数（可选）**
   - 展开"⚙️ 高级设置"
   - 根据需求调整参数
   - 或使用预设配置

4. **开始识别**
   - 点击"🚀 开始识别"按钮
   - 观察实时输出
   - 如需中断，点击"⏹️ 停止"

5. **查看结果**
   - 切换不同标签页查看结果
   - 在Markdown渲染页查看格式化效果
   - 在提取的图片页下载图片

#### 快速体验

点击"示例文件"下的示例，自动填充参数并开始识别。

## ⚙️ 参数说明

### 模型尺寸预设

| 预设 | Base Size | Image Size | Crop Mode | 适用场景 |
|------|-----------|------------|-----------|----------|
| **Tiny** | 512 | 512 | False | 低分辨率文档，快速识别 |
| **Small** | 640 | 640 | False | 一般文档 |
| **Base** | 1024 | 1024 | False | 标准文档 |
| **Large** | 1280 | 1280 | False | 高分辨率文档 |
| **Gundam (推荐)** | 1024 | 640 | True | 平衡精度和速度 |

### 参数详解

#### Base Size
- **作用**: 控制模型处理的基础分辨率
- **范围**: 512-1280
- **建议**:
  - 简单文档：512-640
  - 复杂文档：1024
  - 高精度需求：1280

#### Image Size
- **作用**: 控制输入图像的处理尺寸
- **范围**: 512-1280
- **建议**: 与Base Size配合使用

#### Crop Mode
- **作用**: 是否启用裁剪模式
- **True**: 将大图裁切成小块处理（Gundam模式）
- **False**: 整图处理
- **建议**:
  - 大尺寸文档：True
  - 小图片：False

## 🎨 界面说明

### 四个标签页详解

#### 1. 📝 Markdown渲染

**特点**:
- 自动渲染所有Markdown格式
- 图片使用base64嵌入，直接显示
- 支持表格、公式、代码块
- 最佳的阅读体验

**适用场景**:
- 阅读识别结果
- 检查文档结构
- 查看提取的图片在原文中的位置

**注意**:
- 大型PDF的图片可能较多，加载需要时间
- 图片以base64嵌入，复制时会包含大量数据

#### 2. 📄 纯文本

**特点**:
- 移除所有标注信息
- 纯净的文本或Markdown格式
- 支持一键复制

**适用场景**:
- 复制文本到其他应用
- 保存为纯文本文件
- 二次编辑

#### 3. 🔧 原始输出

**特点**:
- 包含完整的模型输出
- 包括 `<|ref|>` 和 `<|det|>` 标签
- 包含坐标信息

**适用场景**:
- 调试和分析
- 了解模型识别细节
- 二次开发和处理

**标签说明**:
```
<|ref|>类型<|/ref|><|det|>[[坐标]]<|/det|>

类型可能包括:
- title: 标题
- text: 正文
- image: 图片
- table: 表格
- equation: 公式
等等...

坐标格式: [[x1, y1, x2, y2]]
坐标范围: 0-999（归一化坐标）
```

#### 4. 🖼️ 提取的图片

**特点**:
- 画廊展示所有提取的图片
- 3列网格布局
- 支持点击查看大图
- 支持单独下载

**文件命名**:
- 单图片: `page0_img0.jpg`, `page0_img1.jpg`
- PDF多页: `page0_img0.jpg`, `page1_img0.jpg`

**保存位置**:
```
gradio_outputs/
└── session_时间戳/
    └── extracted_images/
        ├── page0_img0.jpg
        ├── page0_img1.jpg
        └── ...
```

## 📊 PDF处理

### PDF识别流程

1. **加载PDF**: 转换为图片（DPI=144）
2. **逐页识别**: 显示进度"第 X/Y 页"
3. **提取图片**: 每页的图片自动提取
4. **结果汇总**: 所有页面结果合并显示

### PDF结果格式

```
==================================================
第 1/5 页
==================================================
[第1页的识别内容]

==================================================
第 2/5 页
==================================================
[第2页的识别内容]

...
```

### PDF图片提取

- 所有页面的图片统一编号
- 按页面顺序排列
- 文件名包含页码信息

## 🛠️ 高级功能

### 流式输出原理

使用Python的生成器（Generator）和多线程实现：

```python
def process_stream():
    # 在新线程中运行模型
    threading.Thread(target=run_model).start()

    # 主线程流式输出
    while processing:
        yield current_text  # 实时更新界面
```

### 图片提取原理

1. 解析 `<|ref|>image<|/ref|><|det|>[[坐标]]<|/det|>` 标签
2. 将归一化坐标（0-999）转换为实际像素
3. 使用PIL裁切图片
4. 保存为高质量JPEG

### Base64图片嵌入

```python
# 读取图片
with open(img_path, 'rb') as f:
    img_data = f.read()

# 转换为base64
img_base64 = base64.b64encode(img_data).decode('utf-8')

# 嵌入到HTML
<img src="data:image/jpeg;base64,{img_base64}">
```

## 🎯 使用技巧

### 1. 提高识别精度

- 使用高分辨率的输入图片
- 选择合适的模型尺寸（Base或Large）
- 确保图片清晰，对比度良好
- 避免图片过度压缩

### 2. 加快识别速度

- 使用较小的模型尺寸（Small或Tiny）
- 降低输入图片分辨率
- 启用Crop Mode处理大图

### 3. 批量处理

- 使用PDF格式合并多个页面
- 一次性上传处理
- 结果自动分页显示

### 4. 图片提取优化

- 确保使用 `Convert to Markdown` 模式
- 带有 `<|grounding|>` 的prompt更容易提取图片
- 检查原始输出确认图片是否被识别

## ⚠️ 注意事项

### 系统要求

1. **GPU要求**
   - 至少8GB显存
   - 支持CUDA
   - 推荐使用A100或V100

2. **内存要求**
   - 系统内存至少16GB
   - 处理大型PDF需要更多内存

3. **磁盘空间**
   - 模型约20GB
   - 识别结果会占用额外空间

### 性能优化

1. **首次运行**
   - 会自动下载模型（约20GB）
   - 下载时间取决于网络速度
   - 模型缓存在 `~/.cache/huggingface/`

2. **大型PDF处理**
   - 可能需要较长时间
   - 建议分批处理
   - 注意观察GPU显存使用

3. **输出目录管理**
   - 定期清理 `gradio_outputs/` 目录
   - 每次识别会创建新的session目录
   - 可以手动删除旧的session

### 常见问题

#### Q1: 模型加载失败
**A**:
- 检查网络连接
- 确保有足够的磁盘空间
- 检查CUDA版本兼容性
- 尝试重新下载模型

#### Q2: 内存不足
**A**:
- 降低base_size和image_size参数
- 关闭crop_mode
- 处理较小的图片
- 分批处理PDF

#### Q3: PDF无法识别
**A**:
- 确保安装了pymupdf: `pip install pymupdf`
- 检查PDF是否为扫描件或图片型PDF
- 确认PDF没有加密或损坏

#### Q4: 图片不显示
**A**:
- 确保使用 `Convert to Markdown` 模式
- 检查 `gradio_outputs/` 目录权限
- 查看终端调试信息
- 确认原始输出中有 `<|ref|>image<|/ref|>` 标签

#### Q5: 识别结果不准确
**A**:
- 提高模型尺寸（使用Large预设）
- 使用高分辨率输入
- 确保图片清晰无模糊
- 尝试调整crop_mode设置

## 🔧 自定义配置

### 修改服务器配置

编辑 `gradio_ocr.py` 最后几行：

```python
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # 监听所有网络接口
        server_port=7860,       # 端口号
        share=True              # 是否创建公网链接
    )
```

**参数说明**:
- `server_name`:
  - `"0.0.0.0"`: 允许远程访问
  - `"127.0.0.1"`: 仅本地访问
- `server_port`: 端口号（默认7860）
- `share`: 创建72小时有效的公网链接

### 修改GPU配置

```python
os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 使用的GPU编号
```

多GPU使用:
```python
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'  # 使用GPU 0和1
```

### 修改输出目录

```python
output_base_dir = os.path.join(script_dir, "my_custom_outputs")
```

## 📝 示例文件

项目包含两个示例文件：

1. **示例图片.png**: 演示图片识别
2. **DeepSeek_OCR_paper-p1.pdf**: 演示PDF识别和图片提取

点击示例即可快速体验功能。

## 🐛 问题排查

### 查看日志

所有调试信息都会输出到终端：

```bash
python gradio_ocr.py

# 输出示例:
Loading model...
Model loaded successfully!
开始处理图片: /path/to/image.png
crop_images_from_text: 开始处理页面 0
找到 1 个图片引用
提取的图片数量: 1
```

### 常用命令

```bash
# 检查CUDA
python -c "import torch; print(torch.cuda.is_available())"

# 检查GPU显存
nvidia-smi

# 清理输出目录
rm -rf gradio_outputs/

# 清理模型缓存
rm -rf ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-OCR
```

## 📚 相关资源

- **DeepSeek-OCR GitHub**: https://github.com/deepseek-ai/DeepSeek-OCR
- **Gradio文档**: https://www.gradio.app/docs
- **PyMuPDF文档**: https://pymupdf.readthedocs.io/

## 🙏 致谢

- DeepSeek AI团队提供的OCR模型
- Gradio提供的Web界面框架
- 开源社区的支持

## 📄 许可证

本项目遵循DeepSeek-OCR的开源许可证。

---

**祝使用愉快！如有问题，请查看上述文档或提交Issue。** 🎉

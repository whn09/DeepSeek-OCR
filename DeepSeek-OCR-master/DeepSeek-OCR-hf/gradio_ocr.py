import gradio as gr
from transformers import AutoModel, AutoTokenizer
import torch
import os
import tempfile
from pathlib import Path
import re
from PIL import Image
import fitz  # PyMuPDF
import io
import sys
import threading
import time
from queue import Queue


os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# 初始化模型
print("Loading model...")
model_name = 'deepseek-ai/DeepSeek-OCR'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    _attn_implementation='flash_attention_2',
    trust_remote_code=True,
    use_safetensors=True
)
model = model.eval().cuda().to(torch.bfloat16)
print("Model loaded successfully!")


class StreamCapture:
    """捕获并流式输出标准输出"""
    def __init__(self):
        self.queue = Queue()
        self.captured_text = []
        self.original_stdout = None
        self.is_capturing = False

    def write(self, text):
        """捕获write调用"""
        if text and text.strip():
            self.captured_text.append(text)
            self.queue.put(text)
        # 同时输出到终端
        if self.original_stdout:
            self.original_stdout.write(text)

    def flush(self):
        """flush方法"""
        if self.original_stdout:
            self.original_stdout.flush()

    def start_capture(self):
        """开始捕获"""
        self.original_stdout = sys.stdout
        sys.stdout = self
        self.is_capturing = True
        self.captured_text = []

    def stop_capture(self):
        """停止捕获"""
        if self.is_capturing:
            sys.stdout = self.original_stdout
            self.is_capturing = False
        return ''.join(self.captured_text)


def pdf_to_images(pdf_path, dpi=144):
    """将PDF转换为图片列表"""
    images = []
    pdf_document = fitz.open(pdf_path)

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)

        Image.MAX_IMAGE_PIXELS = None
        img_data = pixmap.tobytes("png")
        img = Image.open(io.BytesIO(img_data))

        images.append(img)

    pdf_document.close()
    return images


def extract_image_refs(text):
    """提取图片引用标签"""
    pattern = r'(<\|ref\|>image<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches


def extract_coordinates(det_text):
    """从det标签中提取坐标"""
    try:
        coords_list = eval(det_text)
        return coords_list
    except:
        return None


def crop_images_from_text(original_image, ocr_text, output_dir, page_idx=0):
    """根据OCR文本中的坐标裁切图片"""
    if original_image is None or ocr_text is None:
        print("crop_images_from_text: 输入为空")
        return []

    print(f"crop_images_from_text: 开始处理页面 {page_idx}")
    print(f"输出目录: {output_dir}")

    # 提取图片引用
    image_refs = extract_image_refs(ocr_text)
    print(f"找到 {len(image_refs)} 个图片引用")

    if not image_refs:
        return []

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 加载原始图片
    if isinstance(original_image, str):
        img = Image.open(original_image)
        print(f"从文件加载图片: {original_image}")
    else:
        img = original_image
        print("使用PIL Image对象")

    image_width, image_height = img.size
    print(f"图片尺寸: {image_width} x {image_height}")

    cropped_images = []

    for idx, (full_match, coords_str) in enumerate(image_refs):
        print(f"\n处理图片引用 {idx+1}: {coords_str}")

        coords_list = extract_coordinates(coords_str)
        if coords_list is None:
            print("  坐标解析失败")
            continue

        print(f"  解析出 {len(coords_list)} 个坐标框")

        # 处理每个坐标框
        for coord_idx, coord in enumerate(coords_list):
            try:
                x1, y1, x2, y2 = coord
                print(f"  框 {coord_idx+1}: 原始坐标 ({x1}, {y1}) -> ({x2}, {y2})")

                # 将归一化坐标转换为实际像素坐标
                px1 = int(x1 / 999 * image_width)
                py1 = int(y1 / 999 * image_height)
                px2 = int(x2 / 999 * image_width)
                py2 = int(y2 / 999 * image_height)

                print(f"  框 {coord_idx+1}: 像素坐标 ({px1}, {py1}) -> ({px2}, {py2})")

                # 裁切图片
                cropped = img.crop((px1, py1, px2, py2))
                print(f"  裁切后尺寸: {cropped.size}")

                # 保存图片
                img_filename = f"page{page_idx}_img{len(cropped_images)}.jpg"
                img_path = os.path.join(output_dir, img_filename)
                cropped.save(img_path, quality=95)
                print(f"  保存到: {img_path}")

                cropped_images.append(img_path)

            except Exception as e:
                print(f"  裁切图片出错: {e}")
                import traceback
                traceback.print_exc()
                continue

    print(f"\n总共裁切了 {len(cropped_images)} 张图片")
    return cropped_images


def clean_ocr_output(text, image_refs_replacement=None):
    """清理OCR输出文本"""
    if text is None:
        return ""

    # 移除或格式化调试信息
    # 匹配开头的调试块：=====================\nBASE: ...\nPATCHES: ...\n=====================
    text = re.sub(r'={5,}\s*BASE:\s+.*?\s+PATCHES:\s+.*?\s+={5,}\s*', '', text, flags=re.DOTALL)

    # 匹配结尾的调试块：==================================================\nimage size: ...\n...==================================================
    text = re.sub(r'={10,}\s*image size:.*?={10,}\s*', '', text, flags=re.DOTALL)

    # 移除grounding标签
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)

    # 如果提供了图片引用替换，则替换image标签
    img_idx = 0
    for match in matches:
        if '<|ref|>image<|/ref|>' in match[0]:
            if image_refs_replacement:
                # 用实际的图片链接替换
                text = text.replace(match[0], image_refs_replacement.format(img_idx), 1)
                img_idx += 1
            else:
                text = text.replace(match[0], '')
        else:
            text = text.replace(match[0], '')

    # 清理特殊字符
    text = text.replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:')
    text = text.replace('\n\n\n\n', '\n\n').replace('\n\n\n', '\n\n')

    # 移除结束标记
    if '<｜end▁of▁sentence｜>' in text:
        text = text.replace('<｜end▁of▁sentence｜>', '')

    return text.strip()


def create_markdown_with_images(ocr_text, image_paths):
    """创建包含图片链接的Markdown文本"""
    if ocr_text is None:
        return ""

    result_text = ocr_text

    # 移除或格式化调试信息
    # 匹配开头的调试块：=====================\nBASE: ...\nPATCHES: ...\n=====================
    result_text = re.sub(r'={5,}\s*BASE:\s+.*?\s+PATCHES:\s+.*?\s+={5,}\s*', '', result_text, flags=re.DOTALL)

    # 匹配结尾的调试块：==================================================\nimage size: ...\n...==================================================
    result_text = re.sub(r'={10,}\s*image size:.*?={10,}\s*', '', result_text, flags=re.DOTALL)

    # 移除grounding标签，但保留图片位置
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, result_text, re.DOTALL)

    img_idx = 0
    for match in matches:
        if '<|ref|>image<|/ref|>' in match[0]:
            # 用Markdown图片语法替换
            if image_paths and img_idx < len(image_paths):
                img_path = image_paths[img_idx]

                # Gradio需要使用HTML img标签来显示本地图片
                # 使用base64编码嵌入图片
                try:
                    import base64
                    with open(img_path, 'rb') as f:
                        img_data = f.read()
                        img_base64 = base64.b64encode(img_data).decode('utf-8')

                    # 检测图片格式
                    img_ext = os.path.splitext(img_path)[1].lower()
                    if img_ext == '.png':
                        mime_type = 'image/png'
                    elif img_ext in ['.jpg', '.jpeg']:
                        mime_type = 'image/jpeg'
                    else:
                        mime_type = 'image/jpeg'

                    # 使用base64嵌入的img标签
                    img_markdown = f'\n\n<img src="data:{mime_type};base64,{img_base64}" alt="提取的图片{img_idx+1}" style="max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; padding: 5px;">\n\n'

                    print(f"图片{img_idx+1}已转换为base64嵌入 (大小: {len(img_base64)} bytes)")

                except Exception as e:
                    print(f"转换图片{img_idx+1}为base64失败: {e}")
                    img_markdown = f'\n\n[图片{img_idx+1}: {os.path.basename(img_path)}]\n\n'

                result_text = result_text.replace(match[0], img_markdown, 1)
                img_idx += 1
            else:
                result_text = result_text.replace(match[0], '\n[图片]\n', 1)
        else:
            # 移除其他标注
            result_text = result_text.replace(match[0], '')

    # 清理特殊字符
    result_text = result_text.replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:')
    result_text = result_text.replace('\n\n\n\n', '\n\n').replace('\n\n\n', '\n\n')

    # 移除结束标记
    if '<｜end▁of▁sentence｜>' in result_text:
        result_text = result_text.replace('<｜end▁of▁sentence｜>', '')

    return result_text.strip()


def create_pdf_markdown_with_images(raw_results_list, all_image_paths):
    """为PDF创建包含图片的Markdown"""
    if not raw_results_list:
        return ""

    # 合并所有原始结果
    combined_text = "\n".join(raw_results_list)

    # 使用create_markdown_with_images处理
    return create_markdown_with_images(combined_text, all_image_paths)


def process_image_ocr_stream(image, prompt_type, base_size, image_size, crop_mode):
    """处理单张图片的OCR - 生成器版本，支持流式输出"""
    if image is None:
        yield "请上传图片", "", "请上传图片", None
        return

    # 设置prompt
    if prompt_type == "Free OCR":
        prompt = "<image>\nFree OCR. "
    else:  # Convert to Markdown
        prompt = "<image>\n<|grounding|>Convert the document to markdown. "

    # 创建持久化的输出目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_base_dir = os.path.join(script_dir, "gradio_outputs")
    session_dir = os.path.join(output_base_dir, f"session_{int(time.time() * 1000)}")
    os.makedirs(session_dir, exist_ok=True)

    try:
        tmpdir = session_dir
        # 保存上传的图片
        temp_image_path = os.path.join(tmpdir, "input_image.png")
        if isinstance(image, str):
            # 如果是文件路径
            temp_image_path = image
        else:
            # 如果是PIL Image或numpy array
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
            image.save(temp_image_path)

        yield "正在处理图片...", "", "正在处理图片...", None

        # 创建流式捕获对象
        stream_capture = StreamCapture()

        # 在新线程中运行模型推理
        result_container = {'result': None, 'error': None}

        def run_inference():
            try:
                stream_capture.start_capture()
                result = model.infer(
                    tokenizer,
                    prompt=prompt,
                    image_file=temp_image_path,
                    output_path=tmpdir,
                    base_size=int(base_size),
                    image_size=int(image_size),
                    crop_mode=crop_mode,
                    save_results=False,
                    test_compress=True
                )
                result_container['result'] = result
            except Exception as e:
                result_container['error'] = e
            finally:
                stream_capture.stop_capture()

        # 启动推理线程
        inference_thread = threading.Thread(target=run_inference)
        inference_thread.start()

        # 流式输出
        accumulated_text = ""
        last_update = time.time()

        while inference_thread.is_alive() or not stream_capture.queue.empty():
            try:
                # 尝试从队列获取新文本
                text = stream_capture.queue.get(timeout=0.1)
                accumulated_text += text

                # 清理后的文本
                cleaned_text = clean_ocr_output(accumulated_text)

                # 每0.1秒更新一次界面（流式阶段暂不显示图片）
                current_time = time.time()
                if current_time - last_update >= 0.1:
                    yield cleaned_text, accumulated_text, cleaned_text + "\n\n*识别中...*", None
                    last_update = current_time

            except:
                # 队列为空，继续等待
                time.sleep(0.05)

        # 等待线程结束
        inference_thread.join()

        # 检查是否有错误
        if result_container['error']:
            import traceback
            error_msg = f"OCR处理出错: {str(result_container['error'])}\n\n{traceback.format_exc()}"
            yield error_msg, "", error_msg, None
            return

        # 获取最终结果
        final_captured = stream_capture.stop_capture()

        # 使用返回值或捕获的输出
        if result_container['result'] is not None and result_container['result'] != "":
            final_result = result_container['result']
        else:
            final_result = final_captured

        if not final_result:
            yield "OCR处理完成，但模型未返回结果。", "", "OCR处理完成，但模型未返回结果。", None
            return

        # 先输出识别完成的提示，图片正在处理中
        cleaned_temp = clean_ocr_output(final_result)
        yield cleaned_temp, final_result, cleaned_temp + "\n\n*正在提取图片...*", None

        # 裁切图片
        images_dir = os.path.join(tmpdir, "extracted_images")
        cropped_image_paths = crop_images_from_text(temp_image_path, final_result, images_dir, page_idx=0)

        print(f"提取的图片数量: {len(cropped_image_paths)}")
        print(f"图片路径: {cropped_image_paths}")

        # 清理输出（纯文本）
        cleaned_result = clean_ocr_output(final_result)

        # 为Markdown创建带图片链接的版本（图片已经裁切完成）
        markdown_result = create_markdown_with_images(final_result, cropped_image_paths)

        print(f"Markdown结果前500字符:\n{markdown_result[:500]}")

        # 创建图片画廊
        gallery_images = cropped_image_paths if cropped_image_paths else None

        # 最终输出，包含图片
        yield cleaned_result, final_result, markdown_result, gallery_images

    except Exception as e:
        import traceback
        error_msg = f"OCR处理出错: {str(e)}\n\n详细错误:\n{traceback.format_exc()}"
        yield error_msg, "", error_msg, None


def process_pdf_ocr_stream(pdf_file, prompt_type, base_size, image_size, crop_mode):
    """处理PDF文件的OCR - 生成器版本，支持流式输出"""
    if pdf_file is None:
        yield "请上传PDF文件", "", "请上传PDF文件", None
        return

    # 设置prompt
    if prompt_type == "Free OCR":
        prompt = "<image>\nFree OCR. "
    else:  # Convert to Markdown
        prompt = "<image>\n<|grounding|>Convert the document to markdown. "

    # 创建持久化的输出目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_base_dir = os.path.join(script_dir, "gradio_outputs")
    session_dir = os.path.join(output_base_dir, f"session_{int(time.time() * 1000)}")
    os.makedirs(session_dir, exist_ok=True)

    try:
        tmpdir = session_dir

        # 将PDF转换为图片
        yield "正在加载PDF...", "", "正在加载PDF...", None
        images = pdf_to_images(pdf_file.name)

        yield f"PDF加载完成，共 {len(images)} 页，开始识别...", "", f"PDF加载完成，共 {len(images)} 页", None

        all_results = []
        all_raw_results = []
        all_cleaned_results = []
        all_cropped_images = []  # 存储所有裁切的图片

        images_base_dir = os.path.join(tmpdir, "extracted_images")

        for idx, img in enumerate(images):
            temp_image_path = os.path.join(tmpdir, f"page_{idx}.png")
            img.save(temp_image_path)

            page_header = f"\n{'='*50}\n第 {idx+1}/{len(images)} 页\n{'='*50}\n"

            # 更新进度
            progress_msg = "\n".join(all_cleaned_results) + page_header + "正在识别..."
            yield progress_msg, "", progress_msg, all_cropped_images if all_cropped_images else None

            # 创建流式捕获对象
            stream_capture = StreamCapture()

            # 在新线程中运行模型推理
            result_container = {'result': None, 'error': None}

            def run_inference():
                try:
                    stream_capture.start_capture()
                    result = model.infer(
                        tokenizer,
                        prompt=prompt,
                        image_file=temp_image_path,
                        output_path=tmpdir,
                        base_size=int(base_size),
                        image_size=int(image_size),
                        crop_mode=crop_mode,
                        save_results=False,
                        test_compress=True
                    )
                    result_container['result'] = result
                except Exception as e:
                    result_container['error'] = e
                finally:
                    stream_capture.stop_capture()

            # 启动推理线程
            inference_thread = threading.Thread(target=run_inference)
            inference_thread.start()

            # 流式输出当前页
            page_text = ""
            last_update = time.time()

            while inference_thread.is_alive() or not stream_capture.queue.empty():
                try:
                    text = stream_capture.queue.get(timeout=0.1)
                    page_text += text

                    # 清理后的文本
                    cleaned_page = clean_ocr_output(page_text)

                    # 更新界面
                    current_time = time.time()
                    if current_time - last_update >= 0.1:
                        current_display = "\n".join(all_cleaned_results) + page_header + cleaned_page
                        yield current_display, "", current_display, all_cropped_images if all_cropped_images else None
                        last_update = current_time

                except:
                    time.sleep(0.05)

            # 等待线程结束
            inference_thread.join()

            # 获取最终结果
            final_captured = stream_capture.stop_capture()

            if result_container['result'] is not None and result_container['result'] != "":
                final_result = result_container['result']
            else:
                final_result = final_captured

            if not final_result:
                final_result = f"[第 {idx+1} 页处理失败]"

            # 裁切当前页的图片
            page_cropped_images = crop_images_from_text(temp_image_path, final_result, images_base_dir, page_idx=idx)
            all_cropped_images.extend(page_cropped_images)

            print(f"第 {idx+1} 页提取图片数: {len(page_cropped_images)}")

            # 清理输出
            cleaned_result = clean_ocr_output(final_result)

            all_cleaned_results.append(page_header + cleaned_result)
            all_raw_results.append(f"{page_header}(原始输出)\n{final_result}")

        final_clean = "\n".join(all_cleaned_results)
        final_raw = "\n".join(all_raw_results)

        print(f"总共提取图片数: {len(all_cropped_images)}")
        print(f"图片路径: {all_cropped_images}")

        # 创建包含图片的Markdown版本
        # 对于PDF，我们需要重新构建带图片的Markdown
        final_markdown = create_pdf_markdown_with_images(all_raw_results, all_cropped_images)

        yield final_clean, final_raw, final_markdown, all_cropped_images if all_cropped_images else None

    except Exception as e:
        import traceback
        error_msg = f"PDF处理出错: {str(e)}\n{traceback.format_exc()}"
        yield error_msg, "", error_msg, None


def process_file_stream(file, prompt_type, base_size, image_size, crop_mode):
    """统一处理图片或PDF文件 - 生成器版本"""
    if file is None:
        yield "请上传文件", "", "请上传文件", None
        return

    # 判断文件类型
    file_path = file.name if hasattr(file, 'name') else file
    file_ext = Path(file_path).suffix.lower()

    if file_ext == '.pdf':
        yield from process_pdf_ocr_stream(file, prompt_type, base_size, image_size, crop_mode)
    elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
        # 对于图片文件,直接使用路径
        yield from process_image_ocr_stream(file_path, prompt_type, base_size, image_size, crop_mode)
    else:
        yield "不支持的文件格式,请上传图片(jpg/png/bmp等)或PDF文件", "", "不支持的文件格式", None


# 创建Gradio界面
with gr.Blocks(title="DeepSeek OCR", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔍 DeepSeek OCR 文档识别")
    gr.Markdown("支持上传图片或PDF文件进行OCR识别，实时流式输出，支持Markdown渲染")

    with gr.Row():
        with gr.Column(scale=1):
            # 文件上传
            file_input = gr.File(
                label="📁 上传图片或PDF文件",
                file_types=[".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".pdf"]
            )

            # OCR参数设置
            prompt_type = gr.Radio(
                choices=["Free OCR", "Convert to Markdown"],
                value="Convert to Markdown",
                label="🎯 OCR模式"
            )

            with gr.Accordion("⚙️ 高级设置", open=False):
                base_size = gr.Slider(
                    minimum=512,
                    maximum=1280,
                    value=1024,
                    step=64,
                    label="Base Size (基础尺寸)"
                )
                image_size = gr.Slider(
                    minimum=512,
                    maximum=1280,
                    value=640,
                    step=64,
                    label="Image Size (图像尺寸)"
                )
                crop_mode = gr.Checkbox(
                    value=True,
                    label="Crop Mode (裁剪模式)"
                )

                gr.Markdown("""
                **模型尺寸预设参考:**
                - Tiny: base_size=512, image_size=512, crop_mode=False
                - Small: base_size=640, image_size=640, crop_mode=False
                - Base: base_size=1024, image_size=1024, crop_mode=False
                - Large: base_size=1280, image_size=1280, crop_mode=False
                - Gundam (推荐): base_size=1024, image_size=640, crop_mode=True
                """)

            # 执行按钮
            submit_btn = gr.Button("🚀 开始识别", variant="primary", size="lg")
            stop_btn = gr.Button("⏹️ 停止", variant="stop", size="lg")

        with gr.Column(scale=2):
            # 结果显示
            with gr.Tabs():
                with gr.Tab("📝 Markdown渲染"):
                    output_markdown = gr.Markdown(
                        label="Markdown渲染结果",
                        value="等待识别结果..."
                    )

                with gr.Tab("📄 纯文本"):
                    output_clean = gr.Textbox(
                        label="OCR识别结果（已清理）",
                        lines=30,
                        max_lines=50,
                        show_copy_button=True
                    )

                with gr.Tab("🔧 原始输出"):
                    output_raw = gr.Textbox(
                        label="原始OCR输出（包含标注信息）",
                        lines=30,
                        max_lines=50,
                        show_copy_button=True
                    )

                with gr.Tab("🖼️ 提取的图片"):
                    output_gallery = gr.Gallery(
                        label="文档中识别并提取的图片",
                        columns=3,
                        rows=2,
                        height="auto",
                        object_fit="contain",
                        show_download_button=True
                    )

    # 示例
    gr.Markdown("### 📚 示例文件")

    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    example_image = os.path.join(script_dir, "示例图片.png")
    example_pdf = os.path.join(script_dir, "DeepSeek_OCR_paper-p1.pdf")

    # 只添加存在的示例
    examples_list = []
    if os.path.exists(example_image):
        examples_list.append([example_image, "Convert to Markdown", 1024, 640, True])
    if os.path.exists(example_pdf):
        examples_list.append([example_pdf, "Convert to Markdown", 1024, 640, True])

    if examples_list:
        gr.Examples(
            examples=examples_list,
            inputs=[file_input, prompt_type, base_size, image_size, crop_mode],
            label="点击示例快速体验"
        )

    gr.Markdown("### 💡 使用提示")
    gr.Markdown("""
    - 识别过程中会**实时流式显示**结果
    - **Markdown渲染**标签页可以看到格式化后的效果
    - **纯文本**标签页可以复制文本内容
    - **原始输出**标签页包含模型的原始标注信息
    - **提取的图片**标签页显示文档中识别到的所有图片，支持下载
    - 支持PDF多页文档，会逐页识别并显示进度
    """)

    # 绑定事件
    submit_event = submit_btn.click(
        fn=process_file_stream,
        inputs=[file_input, prompt_type, base_size, image_size, crop_mode],
        outputs=[output_clean, output_raw, output_markdown, output_gallery]
    )

    # 停止按钮（取消当前任务）
    stop_btn.click(fn=None, inputs=None, outputs=None, cancels=[submit_event])


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )

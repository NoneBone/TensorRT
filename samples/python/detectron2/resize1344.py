from PIL import Image
import os

def resize_image(image_path, target_size=(1344, 1344)):
    """
    将输入图像resize到指定尺寸，并保存到同一目录，添加尺寸后缀
    
    参数:
        image_path: 输入图像的路径
        target_size: 目标尺寸 (width, height)，默认 (1344, 1344)
    
    返回:
        保存后的文件路径
    """
    # 打开图像
    img = Image.open(image_path)
    
    # Resize图像
    resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
    
    # 构造输出文件路径
    base_dir = os.path.dirname(image_path)
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    suffix = f"_{target_size[0]}x{target_size[1]}"
    output_filename = f"{name}{suffix}{ext}"
    output_path = os.path.join(base_dir, output_filename)
    
    # 保存图像
    resized_img.save(output_path, quality=95, subsampling=0)
    
    return output_path

if __name__ == "__main__":
    # 输入图像路径
    input_image = "/root/cys/PROJECT/00-COMMON/DEMO/05-DT2/demo/input1.jpg"
    
    # 目标尺寸
    target_width = 1344
    target_height = 1344
    
    # 调用函数进行resize
    try:
        output_path = resize_image(input_image, (target_width, target_height))
        print(f"图像已成功resize并保存至: {output_path}")
    except Exception as e:
        print(f"处理图像时发生错误: {e}")
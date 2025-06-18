import os
import argparse
from PIL import Image

def is_image_file(filename):
    return filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))

def get_image_size(path):
    try:
        with Image.open(path) as img:
            return img.size
    except Exception:
        return None

def process_group(group_path, group_name, output_dir):
    face_list = []
    body_list = []
    face_small = []
    body_small = []
    exception = []

    # 确保组路径是绝对路径
    group_abs_path = os.path.abspath(group_path)

    for folder in os.listdir(group_abs_path):
        folder_path = os.path.join(group_abs_path, folder)
        if not os.path.isdir(folder_path):
            continue

        for part in ['face', 'body']:
            part_path = os.path.join(folder_path, part)
            if not os.path.isdir(part_path):
                continue

            # 获取图片文件列表 - 使用绝对路径
            images = [f for f in os.listdir(part_path) if is_image_file(f)]
            image_paths = [os.path.abspath(os.path.join(part_path, f)) for f in images]
            count = len(image_paths)

            # 检查数量异常
            if count < 10 or count > 100:
                exception.append(f"{os.path.abspath(part_path)} (图片数量: {count})")
                continue

            # 检查分辨率
            small_imgs = []
            for img_path in image_paths:
                size = get_image_size(img_path)
                if size is None:
                    continue
                w, h = size
                if w < 50 or h < 50:
                    small_imgs.append(img_path)

            if small_imgs:
                if part == 'face':
                    face_small.extend(small_imgs)
                else:
                    body_small.extend(small_imgs)
                continue

            # 正常图片
            if part == 'face':
                face_list.extend(image_paths)
            else:
                body_list.extend(image_paths)

    # 写入文件
    # prefix = group_name.replace('group_', '')
    prefix = group_name
    # 为每个 group 创建单独的文件夹
    group_output_dir = os.path.join(output_dir, prefix)
    if not os.path.exists(group_output_dir):
        os.makedirs(group_output_dir)

    # 使用前缀命名文件
    with open(os.path.join(group_output_dir, "face_small.txt"), 'w', encoding='utf-8') as f:
        f.write('\n'.join(face_small))
    with open(os.path.join(group_output_dir, "body_small.txt"), 'w', encoding='utf-8') as f:
        f.write('\n'.join(body_small))
    with open(os.path.join(group_output_dir, "face_list.txt"), 'w', encoding='utf-8') as f:
        f.write('\n'.join(face_list))
    with open(os.path.join(group_output_dir, "body_list.txt"), 'w', encoding='utf-8') as f:
        f.write('\n'.join(body_list))
    with open(os.path.join(group_output_dir, "exception.txt"), 'w', encoding='utf-8') as f:
        f.write('\n'.join(exception))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="处理数据集中的图片并生成报告")
    parser.add_argument('--data_dir', type=str, default='../data', help='包含所有 group 的数据目录路径')
    parser.add_argument('--output_dir', type=str, default='../result_list', help='输出结果的目录路径')
    parser.add_argument('--group', type=str, help='指定处理的单个 group (例如 group_0)，不指定则处理所有group')
    args = parser.parse_args()

    # 确保使用绝对路径
    data_dir = os.path.abspath(args.data_dir)
    output_dir = os.path.abspath(args.output_dir)

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"数据目录: {data_dir}")
    print(f"输出目录: {output_dir}")

    # 处理指定的单个group或所有group
    if args.group:
        group_path = os.path.join(data_dir, args.group)
        if os.path.isdir(group_path):
            print(f"处理 {args.group}...")
            process_group(group_path, args.group, output_dir)
        else:
            print(f"错误：找不到 {group_path}")
    else:
        # 处理所有group
        for group in os.listdir(data_dir):
            group_path = os.path.join(data_dir, group)
            if os.path.isdir(group_path) and group.startswith('group_'):
                print(f"处理 {group}...")
                process_group(group_path, group, output_dir)

    print("处理完成。")
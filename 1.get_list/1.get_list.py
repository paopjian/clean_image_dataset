import os
import argparse
from PIL import Image
import time
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="处理数据集中的图片并生成报告")
    parser.add_argument('--data_dir', type=str, default='../data', help='包含所有 group 的数据目录路径')
    parser.add_argument('--output_dir', type=str, default='../result_list', help='输出结果的目录路径')
    parser.add_argument('--group', type=str, help='指定处理的单个 group (例如 group_0)，不指定则处理所有group')
    args = parser.parse_args()
    return args

def is_image_file(filename):
    return filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))


def get_image_size(path):
    try:
        with Image.open(path) as img:
            return img.size
    except Exception:
        return None


def write_to_file(file_path, data_list, mode):
    """将数据列表写入文件"""
    if data_list and len(data_list) > 0:
        with open(file_path, mode, encoding='utf-8') as f:
            f.write('\n'.join(data_list) + '\n')


def process_group(group_path, group_dir, output_dir):
    start_time = time.time()
    face_list = []
    body_list = []
    face_small = []
    body_small = []
    face_exception = []
    body_exception = []

    # 新增变量记录图片数量
    face_count = 0
    body_count = 0

    # 记录每个folder的各类图片计数
    folder_counts = {}

    # 写入文件
    prefix = group_dir
    # 为每个 group 创建单独的文件夹
    group_output_dir = os.path.join(output_dir, prefix)
    if not os.path.exists(group_output_dir):
        os.makedirs(group_output_dir)
    else:
        # 目录已经存在,直接跳过
        print(f"{group_dir} 目录已存在，跳过处理。")
        return 1

    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    temp_log_dir = os.path.join(script_dir, "temp_log")
    os.makedirs(temp_log_dir, exist_ok=True)

    # 确保组路径是绝对路径
    group_abs_path = os.path.abspath(group_path)

    with tqdm(os.listdir(group_abs_path), desc=f"{group_dir}",
              total=len(os.listdir(group_abs_path)),
              leave=True,
              file=open(os.path.join(temp_log_dir, f"progress_{group_dir}.log"), 'w', encoding='utf-8')) as pbar:
        for folder in pbar:
            folder_path = os.path.join(group_abs_path, folder)
            if not os.path.isdir(folder_path):
                continue

            # 初始化当前folder的计数
            folder_counts[folder] = {
                'face_list': 0,
                'face_small': 0,
                'face_exception': 0,
                'body_list': 0,
                'body_small': 0,
                'body_exception': 0
            }

            for part in ['face', 'body']:
                part_path = os.path.join(folder_path, part)
                if not os.path.isdir(part_path):
                    continue

                # 获取图片文件列表 - 使用绝对路径
                images = [f for f in os.listdir(part_path) if is_image_file(f)]
                image_paths = [os.path.abspath(os.path.join(part_path, f)) for f in images]
                count = len(image_paths)

                # 检查数量异常 - 根据类型分别记录异常
                if count < 10 or count > 100:
                    exception_msg = f"{os.path.abspath(part_path)} (图片数量: {count})"
                    if part == 'face':
                        face_exception.append(exception_msg)
                        folder_counts[folder]['face_exception'] = count  # 记录实际异常数量
                    else:
                        body_exception.append(exception_msg)
                        folder_counts[folder]['body_exception'] = count  # 记录实际异常数量
                    # continue

                # 检查分辨率
                small_imgs = []
                for img_path in image_paths[:]:  # 使用副本进行迭代
                    size = get_image_size(img_path)
                    if size is None:
                        image_paths.remove(img_path)  # 从原列表中移除无法获取尺寸的图片
                        continue
                    w, h = size
                    if w < 50 or h < 50:
                        small_imgs.append(img_path)
                        image_paths.remove(img_path)  # 从原列表中移除小图片

                # 将小图片添加到对应列表
                if small_imgs:
                    if part == 'face':
                        face_small.extend(small_imgs)
                        folder_counts[folder]['face_small'] = len(small_imgs)
                    else:
                        body_small.extend(small_imgs)
                        folder_counts[folder]['body_small'] = len(small_imgs)

                # 将正常尺寸的图片添加到对应列表
                if image_paths:  # 仅当有正常尺寸的图片时添加
                    if part == 'face':
                        face_list.extend(image_paths)
                        face_count += len(image_paths)
                        folder_counts[folder]['face_list'] = len(image_paths)
                    else:
                        body_list.extend(image_paths)
                        body_count += len(image_paths)
                        folder_counts[folder]['body_list'] = len(image_paths)
            write_to_file(os.path.join(group_output_dir, "face_small.txt"), face_small, 'a')
            write_to_file(os.path.join(group_output_dir, "body_small.txt"), body_small, 'a')
            write_to_file(os.path.join(group_output_dir, "face_list.txt"), face_list, 'a')
            write_to_file(os.path.join(group_output_dir, "body_list.txt"), body_list, 'a')
            write_to_file(os.path.join(group_output_dir, "face_exception.txt"), face_exception, 'a')
            write_to_file(os.path.join(group_output_dir, "body_exception.txt"), body_exception, 'a')
            face_list = []
            body_list = []
            face_small = []
            body_small = []
            face_exception = []
            body_exception = []

    # 写入每个folder的统计信息
    with open(os.path.join(group_output_dir, "folder_counts.txt"), 'w', encoding='utf-8') as f:
        f.write("folder,face_list,face_small,face_exception,body_list,body_small,body_exception\n")  # CSV头部
        for folder, counts in folder_counts.items():
            f.write(f"{folder},{counts['face_list']},{counts['face_small']},{counts['face_exception']},"
                    f"{counts['body_list']},{counts['body_small']},{counts['body_exception']}\n")

    # # 使用前缀命名文件
    # with open(os.path.join(group_output_dir, "face_small.txt"), 'w', encoding='utf-8') as f:
    #     f.write('\n'.join(face_small))
    # with open(os.path.join(group_output_dir, "body_small.txt"), 'w', encoding='utf-8') as f:
    #     f.write('\n'.join(body_small))
    # with open(os.path.join(group_output_dir, "face_list.txt"), 'w', encoding='utf-8') as f:
    #     f.write('\n'.join(face_list))
    # with open(os.path.join(group_output_dir, "body_list.txt"), 'w', encoding='utf-8') as f:
    #     f.write('\n'.join(body_list))
    # with open(os.path.join(group_output_dir, "face_exception.txt"), 'w', encoding='utf-8') as f:
    #     f.write('\n'.join(face_exception))
    # with open(os.path.join(group_output_dir, "body_exception.txt"), 'w', encoding='utf-8') as f:
    #     f.write('\n'.join(body_exception))

    elapsed_time = time.time() - start_time
    print(f"完成处理 {group_dir}, 耗时: {elapsed_time:.2f}秒")
    print(f"Face图片: {face_count}张, Body图片: {body_count}张")


if __name__ == "__main__":
    args = parse_args()

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

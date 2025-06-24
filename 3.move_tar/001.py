import os
import argparse
import tarfile
import concurrent.futures
import time
from tqdm import tqdm
from pathlib import Path
from io import BytesIO
import re


def parse_args():
    parser = argparse.ArgumentParser(description="从多个tar.gz文件中提取并合并符合条件的图片")
    parser.add_argument('--data_dir', type=str, default='../result_0', help='包含tar.gz文件和face_detect.txt的目录')
    parser.add_argument('--input_dir', type=str, default='../result_2', help='包含tar.gz文件和face_detect.txt的目录')
    parser.add_argument('--output_dir', type=str, default='../result_3', help='输出合并后tar.gz文件的目录')
    parser.add_argument('--group', default=None, type=str,
                        help='指定处理的单个group (例如 group_0)，不指定则处理所有group')
    parser.add_argument('--workers', type=int, default=8, help='并行处理的线程数')
    args = parser.parse_args()
    return args


def read_face_detect_file(file_path):
    """
    读取face_detect.txt文件，返回有效图片名称的集合
    文件格式例如: group_0_0.tar.gz:path/to/image.jpg x1 y1 x2 y2 x3 y3 x4 y4 x5 y5 score
    仅提取图片路径部分
    """
    valid_images = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.split(' ')
                name_path = line[0]
                name_path = name_path.split(':')
                tar_name = name_path[0]
                file_name = name_path[1]
                if tar_name not in valid_images:
                    valid_images[tar_name] = set()
                valid_images[tar_name].add(file_name)
    except Exception as e:
        print(f"读取文件错误 {file_path}: {str(e)}")
        return {}

    print(f"从 {file_path} 成功读取了 {sum(len(images) for images in valid_images.values())} 个有效图片路径")
    return valid_images


def process_tar_file(tar_path, valid_images_dict, output_tar, group_name):
    """
    处理单个tar.gz文件，提取有效图片到输出tar.gz
    """
    tar_name = os.path.basename(tar_path)
    if tar_name not in valid_images_dict:
        print(f"{tar_name} 在face_detect.txt中没有对应项，跳过")
        return 0

    valid_images = valid_images_dict[tar_name]
    extracted_count = 0

    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            for member in tar.getmembers():
                if member.name in valid_images:
                    # 提取图片内容
                    f = tar.extractfile(member)
                    if f is None:
                        continue

                    # 读取图片数据
                    img_data = f.read()

                    # 创建新的TarInfo对象，保持原始名称
                    # 可以选择使用group名称作为前缀，以避免不同组之间的名称冲突
                    # new_name = f"{group_name}/{member.name}"
                    tar_info = tarfile.TarInfo(name=member.name)
                    tar_info.size = len(img_data)

                    # 添加到输出tar文件
                    output_tar.addfile(tar_info, BytesIO(img_data))
                    extracted_count += 1
    except Exception as e:
        print(f"处理文件错误 {tar_path}: {str(e)}")
        return 0

    return extracted_count


def process_group(data_dir, group_path, output_dir, workers):
    """
    处理一个group目录下的所有tar.gz文件
    """
    start_time = time.time()
    group_name = os.path.basename(group_path)
    data_dir = os.path.join(data_dir, group_name)

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 读取face_detect.txt文件
    face_detect_path = os.path.join(group_path, "face_detect.txt")
    if not os.path.exists(face_detect_path):
        print(f"错误：{face_detect_path} 不存在")
        return False

    # 获取有效图片集合
    valid_images_dict = read_face_detect_file(face_detect_path)
    if not valid_images_dict:
        print(f"在 {face_detect_path} 中未找到有效图片路径")
        return False

    # 获取目录中的所有tar.gz文件
    tar_files = [f for f in os.listdir(data_dir) if f.endswith('.tar.gz')]
    if not tar_files:
        print(f"在 {group_path} 中未找到tar.gz文件")
        return False

    # 创建输出的大tar.gz文件
    output_tar_path = os.path.join(output_dir, f"{group_name}_merged.tar.gz")

    total_extracted = 0
    with tarfile.open(output_tar_path, 'w:gz') as output_tar:
        # 使用线程池并行处理tar文件
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            futures = []
            for tar_file in tar_files:
                tar_path = os.path.join(data_dir, tar_file)
                futures.append(
                    executor.submit(process_tar_file, tar_path, valid_images_dict, output_tar, group_name)
                )

            # 显示进度条
            for future in tqdm(concurrent.futures.as_completed(futures),
                               total=len(futures),
                               desc=f"处理 {group_name}"):
                total_extracted += future.result()

    elapsed_time = time.time() - start_time
    print(f"完成处理 {group_name}，共提取 {total_extracted} 张图片，耗时: {elapsed_time:.2f}秒")
    print(f"结果已保存至: {output_tar_path}")

    return True


def main():
    args = parse_args()

    # 确保使用绝对路径
    data_dir = os.path.abspath(args.data_dir)
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")

    # 处理指定的单个group或所有group
    if args.group:
        group_path = os.path.join(input_dir, args.group)
        if os.path.isdir(group_path):
            print(f"处理 {args.group}...")
            process_group(data_dir, group_path, output_dir, args.workers)
        else:
            print(f"错误：找不到 {group_path}")
    else:
        # 处理所有group目录
        for item in os.listdir(input_dir):
            group_path = os.path.join(input_dir, item)
            if os.path.isdir(group_path) and item.startswith('group_'):
                print(f"处理 {item}...")
                process_group(data_dir, group_path, output_dir, args.workers)

    print("所有处理完成。")


if __name__ == "__main__":
    main()

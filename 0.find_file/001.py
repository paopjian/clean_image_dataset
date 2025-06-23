import os
import argparse
from PIL import Image
import time
from tqdm import tqdm
from concurrent.futures import as_completed
import concurrent.futures
import cv2
import pickle
from io import BytesIO
import tarfile
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="处理数据集中的图片并生成报告")
    parser.add_argument('--data_dir', type=str, default='../data', help='包含所有 group 的数据目录路径')
    parser.add_argument('--output_dir', type=str, default='../test', help='输出结果的目录路径')
    parser.add_argument('--group', default='group_0', type=str,
                        help='指定处理的单个 group (例如 group_0)，不指定则处理所有group')
    args = parser.parse_args()
    return args


def is_image_file(filename):
    return filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))


def write_to_file(file_path, data_list, mode):
    """将数据列表写入文件"""
    if data_list and len(data_list) > 0:
        with open(file_path, mode, encoding='utf-8') as f:
            f.write('\n'.join(data_list) + '\n')


def load_single_image(img_path):
    try:
        path = Path(img_path)
        parent_folder = path.parent.parent.name
        filename = path.name
        img_name = f"{parent_folder}+{filename}"
        with Image.open(img_path) as img:
            img = img.convert("RGB")  # 统一转换为 RGB 格式，避免后续保存出错
            return [img_name, img.copy()]
    except Exception as e:
        print(f"读取图像错误 {img_path}: {str(e)}")
        return None


def load_image(folder_path):
    folder_path = os.path.join(folder_path, 'face')
    if not os.path.exists(folder_path):
        print(f"目录不存在: {folder_path}")
        return []
    image_paths = [
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if is_image_file(f)
    ]
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as img_executor:
        future_to_name = {
            img_executor.submit(load_single_image, img_path): img_path
            for img_path in image_paths
        }

        for future in concurrent.futures.as_completed(future_to_name):
            result = future.result()
            if result:
                results.append(result)

    return results


def add_image_to_tar(tar, image_data):
    img = image_data[1]
    img_data = BytesIO()
    img.save(img_data, format="JPEG")
    img_bytes = img_data.getvalue()
    tar_info = tarfile.TarInfo(name=image_data[0])
    tar_info.size = len(img_bytes)
    tar.addfile(tarinfo=tar_info, fileobj=BytesIO(img_bytes))


def save_image_tar(image_list, output_dir, prefix, index):
    with tarfile.open(os.path.join(output_dir, f"{prefix}_{index}.tar.gz"), "w:gz", format=tarfile.PAX_FORMAT) as tar:
        for image_data in image_list:
            add_image_to_tar(tar, image_data)


def process_group(group_path, group_dir, output_dir):
    start_time = time.time()

    # 写入文件
    prefix = group_dir
    # 为每个 group 创建单独的文件夹
    group_output_dir = os.path.join(output_dir, prefix)
    if not os.path.exists(group_output_dir):
        os.makedirs(group_output_dir)
    else:
        pass
        # 目录已经存在,直接跳过
        print(f"{group_dir} 目录已存在，跳过处理。")
        return 1

    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    temp_log_dir = os.path.join(script_dir, "temp_log")
    os.makedirs(temp_log_dir, exist_ok=True)

    # 确保组路径是绝对路径
    group_abs_path = os.path.abspath(group_path)
    # 获取文件夹列表并排序
    folder_list = os.listdir(group_abs_path)
    folder_list = sorted(folder_list)
    folder_list = [os.path.join(group_abs_path, folder) for folder in folder_list if
                   os.path.isdir(os.path.join(group_abs_path, folder))]
    total_folders = len(folder_list)

    # 使用tqdm创建进度条
    with tqdm(total=len(folder_list), desc=f"{group_dir}",
              leave=True,
              file=open(os.path.join(temp_log_dir, f"progress_{group_dir}.log"), 'w', encoding='utf-8')) as pbar:

        # 使用线程池执行并发加载图像
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            # 提交所有图像加载任务
            future_to_path = {
                executor.submit(load_image, folder_path): folder_path for folder_path in folder_list
            }
            index = 0
            results = []

            for future in as_completed(future_to_path):
                pbar.update(1)
                results.append(future.result())
                results_list = [item for sublist in results for item in sublist]
                if len(results) > 200 or len(results_list) > 10000:
                    save_image_tar(results_list, group_output_dir, prefix, index)
                    index += 1
                    results = []
                    results_list = []
            # 最后一次保存
            if len(results) > 0:
                save_image_tar(results_list, group_output_dir, prefix, index)
                index += 1
                results = []

    pass
    elapsed_time = time.time() - start_time
    print(f"完成处理 {group_dir}, 耗时: {elapsed_time:.2f}秒")
    # print(f"Face图片: {face_count}张, Body图片: {body_count}张")


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
        data_dir_list = os.listdir(data_dir)
        data_dir_list = sorted(data_dir_list)
        for group in data_dir_list:
            group_path = os.path.join(data_dir, group)
            if os.path.isdir(group_path) and group.startswith('group_'):
                print(f"处理 {group}...")
                process_group(group_path, group, output_dir)

    print("处理完成。")

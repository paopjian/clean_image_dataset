from __future__ import print_function
import os
import numpy as np
import torch
import argparse
import utils.net as net
from tqdm import tqdm
from utils.dataloader import prepare_dataloader
from collections import defaultdict
import time
from joblib import Parallel, delayed
from multiprocessing import Manager
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def parse_args():
    parser = argparse.ArgumentParser(description='人脸特征提取')
    parser.add_argument('--cuda_id', default=0, type=int, help='CUDA设备ID (默认: 0)')
    parser.add_argument('--input_dir', default='../result_list', type=str, help='结果目录 (默认: ../result)')
    parser.add_argument('--output_dir', type=str, default='../result', help='结果目录')
    parser.add_argument('--specific_folder', default='', type=str,
                        help='指定处理的文件夹完整路径 (如果设置，将只处理该文件夹)')
    parser.add_argument('--cpu', action="store_true", default=False, help='使用CPU而非GPU')
    parser.add_argument('--arch', default='ir_101', type=str, help='模型结构')
    parser.add_argument('--model_path', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                             './pretrained/adaface_ir101_webface12m.ckpt'),
                        type=str,
                        help='模型路径')
    parser.add_argument('--batch_size', default='128', type=int, help='批数量')
    parser.add_argument('--fusion_method', type=str, default='pre_norm_vector_add', choices=('average',
                                                                                             'norm_weighted_avg',
                                                                                             'pre_norm_vector_add',
                                                                                             'concat'))
    parser.add_argument('--use_flip_test', type=str, default='True')
    parser.add_argument('--num_gpus', default=4, type=int, help='要使用的GPU数量 (默认: 4)')
    parser.add_argument('--n_jobs', default=None, type=int, help='并行任务数 (默认: GPU数量)')

    return parser.parse_args()


def load_pretrained_model(args):
    # load model and pretrained statedict
    model = net.build_model(args.arch)
    statedict = torch.load(args.model_path)['state_dict']
    model_statedict = {key[6:]: val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model


def l2_norm(input, axis=1):
    """l2 normalize
    """
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output, norm


def infer_images(model, img_paths, landmarks, batch_size, use_flip_test, device, group_dir):
    # print('total images : {}'.format(len(img_paths)))

    dataloader = prepare_dataloader(img_paths, landmarks, batch_size, num_workers=0, image_size=(112, 112))

    model.eval()
    features = []
    norms = []

    script_dir = os.path.dirname(os.path.abspath(__file__))
    temp_log_dir = os.path.join(script_dir, "temp_log")
    os.makedirs(temp_log_dir, exist_ok=True)

    with torch.no_grad():
        with tqdm(dataloader, desc=f"infer {group_dir}",
                  total=len(dataloader),
                  leave=True,
                  file=open(os.path.join(temp_log_dir, f"progress_{group_dir}.log"), 'w', encoding='utf-8')) as pbar:
            for images, idx in pbar:
                feature = model(images.to(device))
                if isinstance(feature, tuple):
                    feature, norm = feature
                else:
                    norm = None

                if use_flip_test:
                    # infer flipped image and fuse to make a single feature
                    fliped_images = torch.flip(images, dims=[3])
                    flipped_feature = model(fliped_images.to(device))
                    if isinstance(flipped_feature, tuple):
                        flipped_feature, flipped_norm = flipped_feature
                    else:
                        flipped_norm = None

                    stacked_embeddings = torch.stack([feature, flipped_feature], dim=0)
                    if norm is not None:
                        stacked_norms = torch.stack([norm, flipped_norm], dim=0)
                    else:
                        stacked_norms = None

                    pre_norm_embeddings = stacked_embeddings * stacked_norms
                    fused = pre_norm_embeddings.sum(dim=0)
                    feature, norm = l2_norm(fused, axis=1)

                features.append(feature.cpu().numpy())
                norms.append(norm.cpu().numpy())

    features = np.concatenate(features, axis=0)
    img_feats = np.array(features).astype(np.float32)

    norms = np.concatenate(norms, axis=0)

    assert len(features) == len(img_paths)

    return img_feats, norms


def process_folder(folder_path, gpu_pool, args, output_dir, batch_size, use_flip_test_str):
    """
    处理单个文件夹的函数，从GPU池中获取可用GPU
    """
    # 从GPU池中获取可用GPU ID
    cuda_id = gpu_pool.get()
    print(f"在 GPU {cuda_id} 上处理文件夹: {folder_path}")

    try:
        # 设置参数
        use_cpu = False
        use_flip_test = use_flip_test_str.lower() == 'true'

        # 设置设备
        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            torch.cuda.set_device(cuda_id)

        # 加载模型
        model = load_pretrained_model(args)
        device = torch.device(f"cuda:{cuda_id}")
        model = model.to(device)
        print(f'GPU {cuda_id} 完成模型加载！')

        # 获取文件夹名称作为组名
        group_dir = os.path.basename(folder_path)

        # 创建输出目录
        output_folder_path = os.path.join(output_dir, group_dir)
        os.makedirs(output_folder_path, exist_ok=True)

        # 读取face_detect.txt文件，这个文件包含已成功检测的人脸图像及其关键点
        face_detect_file = os.path.join(folder_path, "face_detect.txt")
        if not os.path.exists(face_detect_file):
            print(f"GPU {cuda_id}: 未找到文件: {face_detect_file}")
            return 0

        # 读取face_detect文件中的人脸图像路径和关键点信息
        with open(face_detect_file, 'r', encoding='utf-8') as f:
            face_detect_list = f.readlines()

        # print(f"GPU {cuda_id}: 处理组 {group_dir}, 共 {len(face_detect_list)} 张人脸")
        if len(face_detect_list) == 0:
            print(f"组 {group_dir} 中没有人脸数据，跳过处理")
            return 1

        img_paths = []
        landmarks = []
        faceness_scores = []
        for face_line in face_detect_list:
            parts = face_line.strip().split(' ')

            img_paths.append(parts[0])

            lmk = np.array([float(x) for x in parts[1:-1]], dtype=np.float32)
            lmk = lmk.reshape((5, 2))
            landmarks.append(lmk)

            faceness_score = float(parts[11]) if len(parts) > 11 else 1.0
            faceness_scores.append(faceness_score)

        #  检查存在img_input_feats.npy文件
        feature_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp_log',
                                    f'img_input_feats_{group_dir}.npy')
        if os.path.exists(feature_path):
            print(f"已存在特征文件: {feature_path}，直接加载")
            img_input_feats = np.load(feature_path)
        else:
            img_input_feats, _ = infer_images(model=model, img_paths=img_paths, landmarks=landmarks,
                                              batch_size=batch_size, use_flip_test=use_flip_test, device=device,
                                              group_dir=group_dir)
            np.save(feature_path, img_input_feats)
        # features_normalized = img_input_feats * norms
        features_normalized = img_input_feats
        # 将图片按组号分组
        group_dict = defaultdict(list)
        for i, img_path in enumerate(img_paths):
            # 从路径中提取组号（倒数第三个部分）
            path_parts = img_path.split(os.sep)
            group_id = path_parts[-3]  # 例如从 J:\work\clean\data\group_0\0000045\face\014.jpg 提取 0000045
            group_dict[group_id].append((i, img_path, features_normalized[i]))

        # print(f"GPU {cuda_id}: 共找到 {len(group_dict)} 个不同的组")

        # 为每个组分别处理，创建组目录
        for group_id, group_items in group_dict.items():
            # print(f"GPU {cuda_id}: 处理组 {group_id}，共 {len(group_items)} 张图片")
            if len(group_items) < 2:
                continue  # 如果组内只有一张图片，跳过

            # 为当前组创建目录（在输出目录中）
            group_folder = os.path.join(output_folder_path, group_id)
            os.makedirs(group_folder, exist_ok=True)

            # 创建该组的输出文件路径（在输出目录中）
            group_images_file = os.path.join(group_folder, "related_images.txt")
            unrelated_images_file = os.path.join(group_folder, "unrelated_images.txt")
            # 创建相似度记录文件
            related_similarity_file = os.path.join(group_folder, "related_similarity.txt")
            unrelated_similarity_file = os.path.join(group_folder, "unrelated_similarity.txt")

            # 为该组内的所有图片提取特征
            group_indices = [item[0] for item in group_items]
            group_paths = [item[1] for item in group_items]
            group_features = np.array([item[2] for item in group_items])

            # 计算该组内的相似度矩阵
            group_features_tensor = torch.from_numpy(group_features)
            # 对特征进行L2归一化，确保余弦相似度计算正确
            group_features_normalized = torch.nn.functional.normalize(group_features_tensor, p=2, dim=1)
            # 计算余弦相似度矩阵
            group_similarity = group_features_normalized @ group_features_normalized.T

            # 计算每张图片的相似图片数量（相似度大于0.2的）
            similar_count = defaultdict(int)
            for i in range(len(group_items)):
                for j in range(len(group_items)):
                    if i != j:
                        similarity = group_similarity[i, j].item()
                        if similarity >= 0.2:
                            similar_count[i] += 1

            # 找出具有最多相似图片的那张图片作为基准
            if not similar_count:  # 如果没有相似图片，继续下一组
                continue

            base_idx = max(similar_count, key=similar_count.get)
            base_path = group_paths[base_idx]

            # print(f"GPU {cuda_id}: 组 {group_id} 的基准图片是 {base_path}，有 {similar_count[base_idx]} 张相似图片")

            # 将图片分为相关和不相关两组
            related_images = []
            unrelated_images = []
            related_similarities = []
            unrelated_similarities = []

            # 将基准图片添加到相关图片列表(与自身的相似度为1.0)
            base_filename = os.path.basename(base_path)
            related_images.append(base_filename)
            related_similarities.append((base_filename, 1.0))

            # 将其他图片与基准图片比较
            for i, path in enumerate(group_paths):
                if i == base_idx:
                    continue  # 跳过基准图片自身

                similarity = group_similarity[base_idx, i].item()
                filename = os.path.basename(path)

                if similarity >= 0.2:  # 相似度 >= 0.2 的都归为相关图片
                    related_images.append(filename)
                    related_similarities.append((filename, similarity))
                else:
                    unrelated_images.append(filename)
                    unrelated_similarities.append((filename, similarity))

            # 计算相关图片的平均相似度
            avg_related_similarity = 0.0
            if len(related_similarities) > 1:  # 排除基准图片自身
                avg_related_similarity = sum(sim for _, sim in related_similarities[1:]) / (
                        len(related_similarities) - 1)

            # 计算不相关图片的平均相似度
            avg_unrelated_similarity = 0.0
            if len(unrelated_similarities) > 0:
                avg_unrelated_similarity = sum(sim for _, sim in unrelated_similarities) / len(unrelated_similarities)

            # 写入结果到文件 - 只包含文件名列表
            with open(group_images_file, 'w', encoding='utf-8') as f:
                for img in related_images:
                    f.write(f"{img}\n")

            with open(unrelated_images_file, 'w', encoding='utf-8') as f:
                for img in unrelated_images:
                    f.write(f"{img}\n")

            # 写入相似度信息文件
            with open(related_similarity_file, 'w', encoding='utf-8') as f:
                f.write(f"{avg_related_similarity:.6f}\n")  # 第一行写平均相似度
                for img, sim in related_similarities:
                    f.write(f"{img} {sim:.6f}\n")

            with open(unrelated_similarity_file, 'w', encoding='utf-8') as f:
                f.write(f"{avg_unrelated_similarity:.6f}\n")  # 第一行写平均相似度
                f.write(f"{related_similarities[0][0]} {related_similarities[0][1]:.6f}\n")  # 第二行是基准图片
                for img, sim in unrelated_similarities:
                    f.write(f"{img} {sim:.6f}\n")

            # print(f"GPU {cuda_id}: 组 {group_id} 处理完成")
            # print(f"GPU {cuda_id}: - 相关图片: {len(related_images)}，平均相似度：{avg_related_similarity:.4f}")
            # print(f"GPU {cuda_id}: - 不相关图片: {len(unrelated_images)}，平均相似度：{avg_unrelated_similarity:.4f}")

        return 1
    finally:
        # 无论处理是否成功，都将GPU ID放回池中
        gpu_pool.put(cuda_id)
        print(f"GPU {cuda_id} 已释放回池中")


def main():
    # 解析命令行参数
    args = parse_args()

    # 确定可用的GPU数量
    available_gpus = min(args.num_gpus, torch.cuda.device_count()) if torch.cuda.is_available() else 0

    if available_gpus == 0:
        print("警告：没有可用的GPU，将使用CPU模式")
        args.cpu = True
    else:
        print(f"检测到 {available_gpus} 个可用GPU")

    # 如果使用CPU，不使用多进程
    if args.cpu:
        print("使用CPU模式，将不启用多GPU并行处理")
        # 这里应该添加CPU处理逻辑
        return

    # 设置并行任务数
    n_jobs = args.n_jobs if args.n_jobs is not None else available_gpus

    # 获取输入目录路径
    if args.specific_folder:
        # 直接使用指定的文件夹路径
        folders_to_process = [args.specific_folder]
        print(f"将只处理指定文件夹: {args.specific_folder}")
    else:
        # 如果没有指定文件夹，使用input_dir下的所有子目录
        input_dir = os.path.abspath(args.input_dir)
        print(f"将处理 {input_dir} 下的所有子目录")
        folders_to_process = [os.path.join(input_dir, d) for d in os.listdir(input_dir)
                              if os.path.isdir(os.path.join(input_dir, d))]

    # 过滤掉无效的文件夹
    valid_folders = []
    for folder in folders_to_process:
        if os.path.isdir(folder) and os.path.exists(os.path.join(folder, "face_detect.txt")):
            valid_folders.append(folder)
        else:
            print(f"跳过无效文件夹: {folder}")

    # 确保输出目录存在
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"将处理 {len(valid_folders)} 个有效文件夹")
    if not valid_folders:
        print("没有有效的文件夹可处理，退出")
        return

    # 创建GPU ID池
    manager = Manager()
    gpu_pool = manager.Queue()

    # 初始化GPU池，将所有可用的GPU ID放入池中
    for i in range(available_gpus):
        gpu_pool.put(i)

    # 使用joblib进行并行处理
    start_time = time.time()
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(process_folder)(folder, gpu_pool, args, output_dir, args.batch_size, args.use_flip_test)
        for folder in valid_folders
    )

    end_time = time.time()
    print(f"所有文件夹处理完成，总耗时: {end_time - start_time:.2f} 秒")
    print(f"成功处理文件夹数: {sum(results)}/{len(valid_folders)}")


if __name__ == '__main__':
    main()

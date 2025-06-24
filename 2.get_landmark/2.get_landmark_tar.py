print('第一版, 单线程, 单文件, 每100张图片批量提取一次')
import io
import os
from os.path import exists

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import argparse

from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from tqdm import tqdm
import tarfile

import time


def parse_args():
    parser = argparse.ArgumentParser(description='RetinaFace人脸检测和五点关键点提取')
    parser.add_argument('--cuda_id', default=0, type=int, help='CUDA设备ID (默认: 0)')
    parser.add_argument('--input_dir', default='J:/work/clean/result_0', type=str, help='输入目录,压缩包存储目录')
    parser.add_argument('--output_dir', default='J:/work/clean/result_2', type=str, help='结果目录')
    parser.add_argument('--specific_folder', default='group_0', type=str,
                        help='指定处理的文件夹完整路径 (如果设置，将只处理该文件夹)')  # ../result/group_0
    parser.add_argument('--cpu', action="store_true", default=False, help='使用CPU而非GPU')
    parser.add_argument('--model_path', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                             './weights/retinaface_Resnet50_Final.pth'), type=str,
                        help='模型路径')
    parser.add_argument('--save_image', default='False', type=str, help='保存图片')

    return parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    # print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('加载模型 {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def detect_face(img_raw, net, device, cfg):
    # 读取图像
    # img_raw = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_raw is None:
        # print(f"无法读取图像: {img_path}")
        return None, None

    img = np.float32(img_raw)
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    # 前向传播
    loc, conf, landms = net(img)

    # 后处理
    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / 1
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1 / 1
    landms = landms.cpu().numpy()

    # 过滤低分数
    confidence_threshold = 0.02
    inds = np.where(scores > confidence_threshold)[0]
    if len(inds) == 0:
        return None, None

    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # NMS前保留top-K
    top_k = 5000
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # 执行NMS
    nms_threshold = 0.4
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    dets = dets[keep, :]
    landms = landms[keep]

    # 保留top-K更快的NMS结果
    keep_top_k = 10
    dets = dets[:keep_top_k, :]
    landms = landms[:keep_top_k, :]

    # 如果没有检测到人脸，返回None
    if len(dets) == 0:
        return None, None

    # 返回距离图片中心最近的人脸的五点坐标
    # 计算图片中心点
    image_center_x = im_width / 2
    image_center_y = im_height / 2

    # 计算每个人脸框的中心点到图片中心的距离
    distances = []
    for box in dets[:, :4]:  # 每个box是[x1, y1, x2, y2]
        face_center_x = (box[0] + box[2]) / 2
        face_center_y = (box[1] + box[3]) / 2
        # 计算欧氏距离
        distance = np.sqrt((face_center_x - image_center_x) ** 2 + (face_center_y - image_center_y) ** 2)
        distances.append(distance)

    # 找到距离最小的人脸索引
    best_face_idx = np.argmin(distances)
    best_landms = landms[best_face_idx]
    best_score = dets[best_face_idx, 4]
    return best_landms, best_score


def detect_images(img_list, detect_file, fail_file, file_name, net, device, cfg):
    landmark_list, score_list = [], []
    for img_raw, member_name in img_list:
        # 检测人脸
        landmarks, score = detect_face(img_raw, net, device, cfg)
        landmark_list.append(landmarks)
        score_list.append(score)
    detect_res_file = open(detect_file, 'a', encoding='utf-8')
    fail_res_file = open(fail_file, 'a', encoding='utf-8')

    for landmark, score, (_, member_name) in zip(landmark_list, score_list, img_list):
        # 记录结果
        if landmark is None:
            fail_res_file.write(f"{file_name}:{member_name}\n")
        else:
            landmarks_str = ' '.join([f"{int(point)}" for point in landmark])
            if score is None:
                fail_res_file.write(f"{file_name}:{member_name}\n")
            elif score and score < 0.5:
                fail_res_file.write(f"{file_name}:{member_name} {landmarks_str} {score:.4f}\n")
            else:
                detect_res_file.write(f"{file_name}:{member_name} {landmarks_str} {score:.4f}\n")
    detect_res_file.close()
    fail_res_file.close()


def process_tar_images(folder_path, net, device, cfg, detect_file, fail_file):
    # 遍历文件夹下所有tar.gz文件
    tar_list = os.listdir(folder_path)

    # tar_list = tar_list[:2]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    temp_log_dir = os.path.join(script_dir, "temp_log")
    os.makedirs(temp_log_dir, exist_ok=True)
    group_dir = os.path.basename(folder_path)
    os.makedirs(os.path.join(temp_log_dir, group_dir), exist_ok=True)

    with tqdm(tar_list, desc=f"{os.path.basename(folder_path)}",
              total=len(tar_list),
              leave=True,
              file=open(os.path.join(temp_log_dir, f"progress_{os.path.basename(folder_path)}.log"), 'w',
                        encoding='utf-8')) as pbar:
        for file_name in pbar:
            if not file_name.endswith('.tar.gz'):
                continue
            tar_path = os.path.join(folder_path, file_name)

            img_list = []
            with tarfile.open(tar_path, 'r:gz') as tar:
                member_list = [member for member in tar.getmembers()]

                with tqdm(member_list, desc=f"{file_name}",
                          total=len(member_list),
                          leave=True,
                          file=open(
                              os.path.join(temp_log_dir, group_dir,
                                           f"progress_{os.path.basename(folder_path)}_{file_name}.log"),
                              'w',
                              encoding='utf-8')) as pbar2:
                    for member in pbar2:
                        if not member.isfile():
                            continue
                        # 只处理图片文件
                        if not (member.name.lower().endswith('.jpg') or member.name.lower().endswith('.png')):
                            continue
                        # 读取图片为numpy数组
                        img_file = tar.extractfile(member)
                        if img_file is None:
                            continue
                        img_data = np.frombuffer(img_file.read(), np.uint8)
                        img_raw = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                        img_list.append((img_raw, member.name))
                        if len(img_list) >= 100:
                            detect_images(img_list, detect_file, fail_file, file_name, net, device, cfg)
                            img_list = []

            # 处理剩余的图片
            if img_list:
                detect_images(img_list, detect_file, fail_file, file_name, net, device, cfg)


def main():
    # 解析命令行参数
    args = parse_args()

    # 设置参数
    trained_model = args.model_path
    network = 'resnet50'
    use_cpu = args.cpu

    # 设置设备
    torch.set_grad_enabled(False)
    if not use_cpu and torch.cuda.is_available():
        torch.cuda.set_device(args.cuda_id)

    cfg = {
        'name': 'Resnet50',
        'min_sizes': [[16, 32], [64, 128], [256, 512]],
        'steps': [8, 16, 32],
        'variance': [0.1, 0.2],
        'clip': False,
        'loc_weight': 2.0,
        'gpu_train': True,
        'batch_size': 24,
        'ngpu': 4,
        'epoch': 100,
        'decay1': 70,
        'decay2': 90,
        'image_size': 840,
        'pretrain': None,
        'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
        'in_channel': 256,
        'out_channel': 256
    }

    # 加载网络和模型
    net = RetinaFace(cfg=cfg, phase='test')
    net = load_model(net, trained_model, use_cpu)
    net.eval()
    print(f'完成模型加载！使用设备: {"CPU" if use_cpu else f"CUDA:{args.cuda_id}"}')
    cudnn.benchmark = True
    device = torch.device("cpu" if use_cpu else f"cuda:{args.cuda_id}")
    net = net.to(device)

    os.makedirs(args.output_dir, exist_ok=True)

    # 获取目录路径
    if args.specific_folder:
        # 直接使用指定的文件夹路径，不再与input_dir拼接
        folders_to_process = [os.path.join(args.input_dir, args.specific_folder)]
        print(f"将只处理指定文件夹: {args.specific_folder}")
    else:
        # 如果没有指定文件夹，使用input_dir下的所有子目录
        input_dir = os.path.abspath(args.input_dir)
        print(f"将处理 {input_dir} 下的所有子目录")
        folders_to_process = [os.path.join(input_dir, d) for d in os.listdir(input_dir)
                              if os.path.isdir(os.path.join(input_dir, d))]
    valid_folders = []
    for folder in folders_to_process:
        if os.path.isdir(folder):
            valid_folders.append(folder)
        else:
            print(f"跳过无效文件夹: {folder}")

    for folder_path in valid_folders:
        start_time = time.time()
        if not os.path.isdir(folder_path):
            print(f"跳过非目录: {folder_path}")
            continue

        # 获取文件夹名称作为组名
        group_dir = os.path.basename(folder_path)
        os.makedirs(os.path.join(args.output_dir, group_dir), exist_ok=True)
        if os.path.exists(os.path.join(args.output_dir, group_dir, "face_detect.txt")):
            print(f"跳过已处理的文件夹: {group_dir}")
            continue
        # 创建结果文件
        detect_file = os.path.join(args.output_dir, group_dir, "face_detect.txt")
        fail_file = os.path.join(args.output_dir, group_dir, "face_fail.txt")

        process_tar_images(folder_path, net, device, cfg, detect_file, fail_file)

        print(
            f"组 {group_dir} 处理完成, 结果保存在 {detect_file} {fail_file} 中, 耗时 {time.time() - start_time:.2f} 秒")


if __name__ == '__main__':
    main()

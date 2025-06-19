from __future__ import print_function
import os
import torch
# import torch.backends.cudnn as cudnn
import numpy as np
import argparse
from joblib import Parallel, delayed
import time
from multiprocessing import Manager

from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm


def parse_args():
    parser = argparse.ArgumentParser(description='RetinaFace人脸检测和五点关键点提取')
    parser.add_argument('--cuda_id', default=0, type=int, help='CUDA设备ID (默认: 0)')
    parser.add_argument('--input_dir', default='../result_list', type=str, help='结果目录 (默认: ../result)')
    parser.add_argument('--specific_folder', default='', type=str,
                        help='指定处理的文件夹完整路径 (如果设置，将只处理该文件夹)')  # ../result/group_0
    parser.add_argument('--cpu', action="store_true", default=False, help='使用CPU而非GPU')
    parser.add_argument('--model_path', default='./weights/retinaface_Resnet50_Final.pth', type=str, help='模型路径')
    parser.add_argument('--save_image', default='False', type=str, help='保存图片')
    parser.add_argument('--num_gpus', default=4, type=int, help='要使用的GPU数量 (默认: 4)')
    parser.add_argument('--n_jobs', default=None, type=int, help='并行任务数 (默认: GPU数量)')

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


def detect_face(img_path, net, device, cfg):
    # 读取图像
    img_raw = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_raw is None:
        print(f"无法读取图像: {img_path}")
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


def process_folder(folder_path, gpu_pool, model_path, save_image=False):
    """
    处理单个文件夹的函数，从GPU池中获取可用GPU
    """
    # 从GPU池中获取可用GPU ID
    cuda_id = gpu_pool.get()
    print(f"在 GPU {cuda_id} 上处理文件夹: {folder_path}")

    try:
        # 设置参数
        trained_model = model_path
        use_cpu = False

        # 设置设备
        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            torch.cuda.set_device(cuda_id)

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
        print(f'GPU {cuda_id} 完成模型加载！')
        # cudnn.benchmark = True
        device = torch.device(f"cuda:{cuda_id}")
        net = net.to(device)

        # 获取文件夹名称作为组名
        group_dir = os.path.basename(folder_path)

        # 使用组名构建文件名
        face_list_file = os.path.join(folder_path, "face_list.txt")
        if not os.path.exists(face_list_file):
            print(f"GPU {cuda_id}: 未找到文件: {face_list_file}")
            return 0

        # 创建结果文件
        detect_file = os.path.join(folder_path, "face_detect.txt")
        fail_file = os.path.join(folder_path, "face_fail.txt")

        detect_results = []
        fail_results = []

        # 读取face列表并处理每个图像
        with open(face_list_file, 'r', encoding='utf-8') as f:
            face_list = f.readlines()

        print(f"GPU {cuda_id}: 处理组 {group_dir}, 共 {len(face_list)} 张图像")

        for face_path in face_list:
            face_path = face_path.strip()
            if not face_path:
                continue

            # 检测人脸
            landmarks, score = detect_face(face_path, net, device, cfg)

            # 判断检测结果并记录
            if landmarks is None:
                fail_results.append(face_path)
            else:
                # 置信度检查
                if score < 0.5:
                    fail_results.append(face_path)
                else:
                    # 将五点坐标格式化为字符串
                    landmarks_str = ' '.join([f"{point:.1f}" for point in landmarks])
                    detect_results.append(f"{face_path} {landmarks_str} {score:.4f}")

                    if save_image:
                        text = "{:.4f}".format(score)

                        img_raw = cv2.imread(face_path, cv2.IMREAD_COLOR)
                        if img_raw is None:
                            continue

                        landmarks = list(map(int, landmarks))
                        cv2.putText(img_raw, text, (0, 0),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                        cv2.circle(img_raw, (landmarks[0], landmarks[1]), 1, (0, 0, 255), 4)
                        cv2.circle(img_raw, (landmarks[2], landmarks[3]), 1, (0, 255, 255), 4)
                        cv2.circle(img_raw, (landmarks[4], landmarks[5]), 1, (255, 0, 255), 4)
                        cv2.circle(img_raw, (landmarks[6], landmarks[7]), 1, (0, 255, 0), 4)
                        cv2.circle(img_raw, (landmarks[8], landmarks[9]), 1, (255, 0, 0), 4)

                        os.makedirs('../face_result', exist_ok=True)
                        name = os.path.join('../face_result', os.path.join('_'.join(face_path.split('\\')[-3:])))
                        cv2.imwrite(name, img_raw)

        # 写入检测成功的结果
        with open(detect_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(detect_results))

        # 写入检测失败的结果
        with open(fail_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(fail_results))

        print(f"GPU {cuda_id}: 组 {group_dir} 处理完成 - 成功: {len(detect_results)}, 失败: {len(fail_results)}")
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
        args.cuda_id = -1
        # 使用原始处理逻辑处理所有文件夹
        # ...
        return

    # 设置并行任务数
    n_jobs = args.n_jobs if args.n_jobs is not None else available_gpus

    # 获取目录路径
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
        if os.path.isdir(folder) and os.path.exists(os.path.join(folder, "face_list.txt")):
            valid_folders.append(folder)
        else:
            print(f"跳过无效文件夹: {folder}")

    print(f"将处理 {len(valid_folders)} 个有效文件夹")
    if not valid_folders:
        print("没有有效的文件夹可处理，退出")
        return

    # 保存图片参数转换为布尔值
    save_image = args.save_image.lower() == 'true'

    # 创建GPU ID池
    manager = Manager()
    gpu_pool = manager.Queue()

    # 初始化GPU池，将所有可用的GPU ID放入池中
    for i in range(available_gpus):
        gpu_pool.put(i)

    # 使用joblib进行并行处理
    start_time = time.time()
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(process_folder)(folder, gpu_pool, args.model_path, save_image)
        for folder in valid_folders
    )

    end_time = time.time()
    print(f"所有文件夹处理完成，总耗时: {end_time - start_time:.2f} 秒")
    print(f"成功处理文件夹数: {sum(results)}/{len(valid_folders)}")


if __name__ == '__main__':
    main()

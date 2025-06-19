# 安装流程
```
#安装环境
conda create -n zkj-work python=3.12
conda activate zkj-work
pip3 install scikit-image matplotlib pandas scikit-learn
pip3 install tqdm opencv-python
#pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip3 install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple
# 设置权重文件夹
mkdir 2.get_landmark/weights
mkdir 3.get_features/pretrained
# retinaface_Resnet50_Final.pth 放入 2.get_landmark/weights
# adaface_ir101_webface12m.ckpt 放入 3.get_features/pretrained

# 启动脚本

```

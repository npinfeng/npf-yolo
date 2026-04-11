import torch
from ultralytics import YOLO

def main():
    # 1. 载入模型配置。这里指定你文件夹下的 yolo26n.pt
    # （注：yolo26n.pt 应该是基于yolov8/yolo11架构重命名或定制的权重文件）
    model = YOLO('yolo26n.pt') 
    
    if torch.cuda.is_available():
        device = 0
    else:
        device = 'cpu'
        
    # 注册回调函数，用于明确打印当前进行的是第几个 epoch
    def on_train_epoch_start(trainer):
        print(f"\n" + "="*15 + f" 🚀 当前正在训练第 {trainer.epoch + 1} / {trainer.epochs} 轮 (Epoch) " + "="*15)
        
    model.add_callback("on_train_epoch_start", on_train_epoch_start)

    # 2. 指定数据集的配置文件路径
    # 请确保在训练前已经运行了 prepare_yolo_dataset.py 生成了标准格式数据集和 dataset.yaml
    dataset_yaml = r'f:\npfcode\data\yolo_dataset\dataset.yaml'

    # 3. 开始训练
    # 常用训练参数：
    # epochs: 训练轮数
    # imgsz: 图像输入尺寸
    # batch: 批次大小 (如果显存不足可以调小例如 8, 4)
    # workers: 数据加载的多线程数
    results = model.train(
        data=dataset_yaml, 
        epochs=300, 
        imgsz=640, 
        batch=16, 
        device=device,  # 使用上方判断的 device
        name='chip_defect_detection' # 训练结果将保存在 runs/detect/chip_defect_detection 下
    )

    print("模型训练完毕！")

if __name__ == '__main__':
    main()

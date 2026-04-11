from ultralytics import YOLO

def main():
    model = YOLO(r'E:\projects\npf-yolo\runs\detect\chip_defect_detection\weights\best.pt') 
    
    dataset_yaml = r'E:\data\yolo_dataset\dataset.yaml'
    
    print("开始在验证集上评估...")
    # 指定 project 为当前项目下的 runs/detect
    metrics = model.val(data=dataset_yaml, project=r'E:\projects\npf-yolo\runs\detect')
    
    print(f"评估完成！")
    print(f"验证集 mAP50-95: {metrics.box.map}") # 打印平均精度
    print(f"验证集 mAP50: {metrics.box.map50}")
    print(f"验证集 recall: {metrics.box.mr}")
    print(f"验证集 Precision: {metrics.box.mp}")

    print("\n开始进行图片推理测试...")
    test_source = r'E:\data\yolo_dataset\images\val'
    
    # save=True 会将画好检测框的图片保存到 runs/detect/predict 文件夹下
    # 指定 project 为当前项目下的 runs/detect
    results = model.predict(source=test_source, save=True, save_txt=True, conf=0.25, project=r'E:\projects\npf-yolo\runs\detect')
    
    print("推理测试成功！可视化的结果已保存在 runs/detect/predict 中。")

if __name__ == '__main__':
   
    main()

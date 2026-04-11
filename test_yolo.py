from ultralytics import YOLO

def main():
    model = YOLO(r'F:\npfcode\npf-yolo\runs\detect\chip_defect_detection\weights\best.pt') 
    
    dataset_yaml = r'F:\npfcode\npf-yolo\data\yolo_dataset\dataset.yaml'
    
    print("开始在验证集上评估...")
    metrics = model.val(data=dataset_yaml)
    
    print(f"评估完成！")
    print(f"验证集 mAP50-95: {metrics.box.map}") # 打印平均精度
    print(f"验证集 mAP50: {metrics.box.map50}")
    print(f"验证集 recall: {metrics.box.R}")
    print(f"验证集 Precision: {metrics.box.P}")

    print("\n开始进行图片推理测试...")
    test_source = r'F:\npfcode\npf-yolo\data\yolo_dataset\images\val'
    
    # save=True 会将画好检测框的图片保存到 runs/detect/predict 文件夹下
    results = model.predict(source=test_source, save=True, save_txt=True, conf=0.25)
    
    print("推理测试成功！可视化的结果已保存在 runs/detect/predict 中。")

if __name__ == '__main__':
   
    main()

"""
============================================================
  YOLO 批量自动训练脚本 (Windows)
  
  使用方法:
    1. 在下方 TRAIN_TASKS 列表中添加你的训练任务
    2. 根据需要修改 COMMON_SETTINGS 中的公共训练参数
    3. 运行: python batch_train.py
    
  脚本会自动按顺序完成所有训练任务，出错时跳过并继续下一个
============================================================
"""

import sys
import time
import datetime
import traceback

from ultralytics import YOLO


# ===========================================================================
#  ★★★ 在这里配置你的训练任务 ★★★
#  每个任务只需要设置:
#    - "model":  模型配置文件路径 (.yaml) 或预训练权重 (.pt)
#    - "name":   实验保存名称 (结果会保存在 runs/detect/<name>)
#    - 其他可选参数会覆盖 COMMON_SETTINGS 中的同名参数
# ===========================================================================
TRAIN_TASKS = [
    {
        "model": "ultralytics/cfg/models/26/yolo26.yaml",
        "name": "yolo26_baseline",
    },
    {
        "model": "ultralytics/cfg/models/26/yolo26-dynamic-routing.yaml",
        "name": "yolo26_dynamic_routing",
    },
    {
        "model": "ultralytics/cfg/models/26/yolo26-dynamic-routing_45.yaml",
        "name": "yolo26_dynamic_routing_45",
    },
    # ── 继续添加更多任务 ──────────────────────
    {
        "model": "ultralytics/cfg/models/26/yolo26-dynamic-routing_gate_three.yaml",
        "name": "yolo26_gate_three",
        "epochs": 300,  # 可以为单个任务覆盖公共参数
    },
]


# ===========================================================================
#  ★★★ 公共训练参数 ★★★
#  所有任务共用以下参数，单个任务中的同名参数会覆盖这里的值
# ===========================================================================
COMMON_SETTINGS = {
    "data": r"f:\npfcode\data\yolo_dataset\dataset.yaml",  # 数据集配置文件路径
    "epochs": 300,          # 训练轮数
    "batch": 32,            # 批大小 (根据显存调整, -1 自动)
    "imgsz": 640,           # 输入图像尺寸
    "device": 0,            # GPU 设备 (0, [0,1], 'cpu')          # 数据加载线程数
    "patience": 50,        # 早停耐心值 (0 表示不启用早停)
    "optimizer": "auto",    # 优化器
    "pretrained": True,     # 是否使用预训练权重
    "save": True,           # 保存模型
    "save_period": -1,      # 每N个epoch保存 (-1 关闭)
    "cache": False,         # 缓存图片到内存
    "exist_ok": False,      # 是否覆盖已有实验目录
    "project": "runs/detect_26",  # 项目保存根目录
    "amp": True,            # 混合精度训练
    "cos_lr": False,        # 余弦学习率调度
    "seed": 42,              # 随机种子
    "verbose": True,        # 详细日志
    "plots": True,          # 保存训练曲线图
}


# ===========================================================================
#  以下为脚本逻辑，一般不需要修改
# ===========================================================================

def format_duration(seconds: float) -> str:
    """将秒数格式化为易读的时间字符串"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    else:
        return f"{s}s"


def print_banner(text: str, char: str = "=", width: int = 70):
    """打印分隔横幅"""
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def run_single_task(task: dict, task_index: int, total: int) -> dict:
    """
    执行单个训练任务
    返回: {"name": str, "status": "成功"|"失败", "duration": float, "error": str|None}
    """
    model_path = task.pop("model")
    exp_name = task.get("name", f"exp_{task_index}")

    # 合并公共参数和任务专属参数 (任务参数优先)
    train_args = {**COMMON_SETTINGS, **task}

    print_banner(
        f"任务 [{task_index}/{total}]  模型: {model_path}  名称: {exp_name}",
        char="━",
    )
    print(f"  训练参数:")
    for k, v in train_args.items():
        print(f"    {k}: {v}")
    print()

    start_time = time.time()
    error_msg = None

    try:
        # 加载模型
        model = YOLO(model_path)
        # 开始训练
        model.train(**train_args)

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        print(f"\n  ✗ 训练出错: {e}")

    duration = time.time() - start_time
    status = "成功 ✓" if error_msg is None else "失败 ✗"

    print(f"\n  状态: {status}    耗时: {format_duration(duration)}")
    return {
        "name": exp_name,
        "model": model_path,
        "status": status,
        "duration": duration,
        "error": error_msg,
    }


def main():
    total = len(TRAIN_TASKS)
    if total == 0:
        print("没有配置任何训练任务，请在 TRAIN_TASKS 中添加任务后重试。")
        sys.exit(0)

    print_banner(
        f"YOLO 批量训练  |  共 {total} 个任务  |  "
        f"开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        char="█",
    )

    results = []

    for i, task in enumerate(TRAIN_TASKS, start=1):
        # 深拷贝，避免 pop 影响原数据
        task_copy = dict(task)
        result = run_single_task(task_copy, i, total)
        results.append(result)

    # ── 打印最终汇总 ──────────────────────────
    print_banner("所有训练任务已完成 — 汇总报告", char="█")

    total_time = sum(r["duration"] for r in results)
    success_count = sum(1 for r in results if "成功" in r["status"])
    fail_count = total - success_count

    print(f"\n  总耗时:   {format_duration(total_time)}")
    print(f"  成功:     {success_count}/{total}")
    print(f"  失败:     {fail_count}/{total}")
    print(f"\n  {'序号':<6}{'实验名称':<35}{'状态':<10}{'耗时':<15}")
    print(f"  {'─' * 66}")

    for i, r in enumerate(results, 1):
        print(
            f"  {i:<6}{r['name']:<35}{r['status']:<10}"
            f"{format_duration(r['duration']):<15}"
        )

    if fail_count > 0:
        print(f"\n  ⚠ 以下任务训练失败:")
        for r in results:
            if "失败" in r["status"]:
                print(f"    - {r['name']}: {r['error'].splitlines()[0]}")

    print()


if __name__ == "__main__":
    main()

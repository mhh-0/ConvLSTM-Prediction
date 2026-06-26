

FILE_PATHS = [#文件
    f'./0-4 new-dataset/DN50N12K067B060-0-4-2-{i:05d}'
    for i in range(3912, 4319, 2)
]



INPUT_STEPS = 4     #每次输入

HIDDEN_DIMS = [64, 128]   #隐藏层

KERNEL_SIZE = 3     


GRID_SIZE = (64, 64)



TRAIN_RATIO = 0.8



BATCH_SIZE = 8            

EPOCHS = 300

LR = 5e-4

GRAD_CLIP = 1.0             # 梯度裁剪



LAMBDA_BOUNDARY = 0.2      # 边界加权 MSE 系数

LAMBDA_GRADIENT = 0.01   # 梯度损失系数


# =====================================
# 学习率调度
# =====================================

T_0 = 20                    # CosineAnnealingWarmRestarts 初始周期

T_MULT = 2                  # 每轮周期倍增因子


# =====================================
# 保存路径
# =====================================

MODEL_SAVE_PATH = "convlstm_flow_model.pth"

# 可视化输出
RESULT_FIG_PATH = "convlstm_training_results.png"
FLOW_FIELD_FIG_PATH = "convlstm_flow_field.png"

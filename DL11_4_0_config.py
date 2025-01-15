import os
import torch

class Config:
    # 代理设置
    HTTP_PROXY = 'http://127.0.0.1:7890'
    HTTPS_PROXY = 'http://127.0.0.1:7890'
    
    # 模型配置
    BERT_MODEL_NAME = "bert-base-cased"
    XLMR_MODEL_NAME = "xlm-roberta-base"
    
    # 训练配置
    NUM_EPOCHS = 3
    BATCH_SIZE = 24
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    LOGGING_STEPS = None  # 将在运行时根据数据集大小设置
    
    # 语言配置
    LANGS = ["de", "fr", "it", "en"]
    LANG_FRACS = [0.629, 0.229, 0.084, 0.059]
    
    # 设备配置
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def setup_environment():
        """设置环境变量和代理"""
        os.environ['HTTP_PROXY'] = Config.HTTP_PROXY
        os.environ['HTTPS_PROXY'] = Config.HTTPS_PROXY
        
    # 训练模式配置
    TRAINING_MODE = ["transfer", "joint"] #纯zero-shot transfer "vs" 联合训练
    TRANSFER_MODEL_DIR = "xlmr-finetuned-panx-transfer"
    JOINT_MODEL_DIR = "xlmr-finetuned-panx-joint"
    
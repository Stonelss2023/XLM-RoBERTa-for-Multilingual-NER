from transformers import (
    AutoConfig,
    XLMRobertaForTokenClassification,
    DataCollatorForTokenClassification
)
from DL11_4_0_config import Config

class NERModel:
    def __init__(self, num_labels, id2label, label2id):
        self.config = AutoConfig.from_pretrained(
            Config.XLMR_MODEL_NAME,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id
        )
        
        self.model = XLMRobertaForTokenClassification.from_pretrained(
            Config.XLMR_MODEL_NAME,
            config=self.config
        ).to(Config.DEVICE)
    
    def get_model(self):
        return self.model

    def model_init(self):
        """返回模型初始化函数"""
        return lambda: XLMRobertaForTokenClassification.from_pretrained(
            Config.XLMR_MODEL_NAME,
            config=self.config
        ).to(Config.DEVICE)
import numpy as np
from seqeval.metrics import f1_score, classification_report, precision_score, recall_score
import pandas as pd

"""seqeval.metrics库用于标注任务的评估指标计算"""
class NEREvaluator:
    def __init__(self, id2label):
        """初始化评估器"""
        self.id2label = id2label
    
    def align_predictions(self, predictions, label_ids):
        """对齐预测结果和标签"""
        preds = np.argmax(predictions, axis=2) #在num_labels维度进行选择
        batch_size, seq_len = preds.shape
        # predictions原形状(batch_size, seq_len, num_labels)
        preds_list = [[self.id2label[preds[batch_idx][seq_idx]] 
                      for seq_idx in range(seq_len) 
                      if label_ids[batch_idx][seq_idx] != -100]
                     for batch_idx in range(batch_size)]
        
        labels_list = [[self.id2label[label_ids[batch_idx][seq_idx]]
                       for seq_idx in range(seq_len)
                       if label_ids[batch_idx][seq_idx] != -100]
                      for batch_idx in range(batch_size)]
        
        return preds_list, labels_list

    def compute_metrics(self, eval_pred):
        """计算评估指标, 使用micro平均方式计算指标"""
        predictions, labels = eval_pred #eval_pred是Hugging Face类在Trainer类评估阶段自动传入的参数
        # 这里是元组解包语句。Trainer评估时会自动收集这些东西
        predictions = np.argmax(predictions, axis=2)
        
        # 转换为标签形式
        true_predictions = [
            [self.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        # 计算主要指标
        results = {
            "precision": precision_score(true_labels, true_predictions, average="micro"),
            "recall": recall_score(true_labels, true_predictions, average="micro"),
            "f1": f1_score(true_labels, true_predictions, average="micro")
        }
        
        # 获取详细分类报告
        report = classification_report(true_labels, true_predictions, output_dict=True)
        # 合并结果
        results["classification_report"] = report
        return results #Hugging Face Trainer类期望compute_metrics仅返回一个字典

    def evaluate_lang_performance(self, trainer, dataset, lang=None):
        """评估模型在特定语言上的表现"""
        metrics = trainer.evaluate(dataset["test"])
        
        results = {
            "language": lang,
            "f1": metrics["eval_f1"],
            "precision": metrics["eval_precision"],
            "recall": metrics["eval_recall"]
        }
        
        return results
    
    
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForTokenClassification
import numpy as np
import torch
from typing import Dict, List, Optional
import os

class NERTrainer:
    def __init__(self, model, tokenizer, output_dir, train_dataset=None, eval_dataset=None, training_mode="transfer"):
        """
        初始化 NER 训练器
        
        Args:
            model: 预训练模型
            tokenizer: 分词器
            output_dir: 输出目录
            train_dataset: 训练数据集
            eval_dataset: 评估数据集
            training_mode: 训练模式 ("transfer" 或 "joint")
        """
        self.model = model
        self.tokenizer = tokenizer
        self.training_mode = training_mode
        self.output_dir = output_dir
        
        # 创建数据整理器
        self.data_collator = DataCollatorForTokenClassification(tokenizer)
        
        # 设置训练参数
        self.training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            learning_rate=2e-5,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            logging_dir=os.path.join(output_dir, "logs"),
            logging_steps=100,
            save_total_limit=2,
        )
        
        # 初始化 Trainer
        self.trainer = Trainer(
            model=model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=self.data_collator
        )
        
        # 添加自定义 state 属性来存储训练历史
        self.state = type('TrainingState', (), {'log_history': []})()\

    def train(self):
        """
        训练模型并记录训练历史
        
        Returns:
            训练结果
        """
        print(f"\nStarting {self.training_mode} training...")
        train_result = self.trainer.train()
        
        # 更新训练历史
        self.state.log_history = self.trainer.state.log_history
        
        # 保存最终模型
        self.trainer.save_model(self.output_dir)
        
        # 保存训练状态
        metrics = train_result.metrics
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        self.trainer.save_state()
        
        return train_result

    def evaluate(self, eval_dataset=None):
        """
        评估模型性能
        
        Args:
            eval_dataset: 可选的评估数据集，如果不提供则使用初始化时的评估数据集
            
        Returns:
            评估指标
        """
        if eval_dataset is not None:
            self.trainer.eval_dataset = eval_dataset
            
        print("\nEvaluating model...")
        metrics = self.trainer.evaluate()
        
        # 保存评估指标
        self.trainer.log_metrics("eval", metrics)
        self.trainer.save_metrics("eval", metrics)
        
        return metrics

    def predict(self, test_dataset):
        """
        在测试集上进行预测
        
        Args:
            test_dataset: 测试数据集
            
        Returns:
            预测结果、标签和指标
        """
        print("\nGenerating predictions...")
        predictions, labels, metrics = self.trainer.predict(test_dataset)
        predictions = np.argmax(predictions, axis=2)
        
        return predictions, labels, metrics

    def get_training_history(self):
        """
        获取训练历史记录，并按照合适的格式组织
        
        Returns:
            包含step_history和epoch_history的字典
        """
        history = self.state.log_history
        
        # 初始化历史记录字典
        formatted_history = {
            'step_history': {'steps': [], 'train_loss': []},
            'epoch_history': {
                'train_loss': [], 
                'eval_loss': [],
                'eval_f1': [],
                'eval_precision': [],
                'eval_recall': []
            }
        }
        
        # 整理历史数据
        for entry in history:
            if 'step' in entry:
                formatted_history['step_history']['steps'].append(entry['step'])
                if 'loss' in entry:
                    formatted_history['step_history']['train_loss'].append(entry['loss'])
            
            if 'epoch' in entry:
                if 'loss' in entry:
                    formatted_history['epoch_history']['train_loss'].append(entry['loss'])
                if 'eval_loss' in entry:
                    formatted_history['epoch_history']['eval_loss'].append(entry['eval_loss'])
                if 'eval_f1' in entry:
                    formatted_history['epoch_history']['eval_f1'].append(entry['eval_f1'])
                if 'eval_precision' in entry:
                    formatted_history['epoch_history']['eval_precision'].append(entry['eval_precision'])
                if 'eval_recall' in entry:
                    formatted_history['epoch_history']['eval_recall'].append(entry['eval_recall'])
        
        return formatted_history





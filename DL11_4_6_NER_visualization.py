import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

class Visualizer:
    @staticmethod
    def _set_plt_style():
        """设置matplotlib的基本样式"""
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号
        
    @staticmethod
    def plot_training_history(history, output_path=None):
        """绘制训练历史曲线，包括步级别和轮级别的数据"""
        Visualizer._set_plt_style()
        
        # 创建两个子图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

        # 绘制详细的训练损失曲线（步级别）
        step_history = history.get('step_history', {'steps': [], 'train_loss': []})
        if len(step_history['steps']) > 0 and len(step_history['train_loss']) > 0:
            if len(step_history['steps']) == len(step_history['train_loss']):  # 确保维度匹配
                ax1.plot(step_history['steps'], 
                        step_history['train_loss'],
                        'b-', alpha=0.3, label='Training Loss (Steps)')

        # 绘制epoch级别的损失曲线
        epoch_history = history.get('epoch_history', {
            'train_loss': [], 
            'eval_loss': [],
            'eval_f1': []
        })
        
        # 只在有数据时绘制
        if len(epoch_history['train_loss']) > 0:
            epochs = range(1, len(epoch_history['train_loss']) + 1)
            ax1.plot(epochs, epoch_history['train_loss'], 
                    'b-', linewidth=2, label='Training Loss (Epoch)')
            
            if len(epoch_history['eval_loss']) == len(epoch_history['train_loss']):
                ax1.plot(epochs, epoch_history['eval_loss'], 
                        'r-', linewidth=2, label='Validation Loss')
            
            ax1.set_title('Training and Validation Loss')
            ax1.set_xlabel('Steps/Epochs')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)

            # 只在有F1数据时绘制F1分数曲线
            if len(epoch_history['eval_f1']) > 0 and len(epoch_history['eval_f1']) == len(epochs):
                ax2.plot(epochs, epoch_history['eval_f1'], 'g-', label='F1 Score')
                ax2.set_title('Validation F1 Score')
                ax2.set_xlabel('Epochs')
                ax2.set_ylabel('F1 Score')
                ax2.legend()
                ax2.grid(True)

        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_language_performance(results, output_path=None):
        """绘制各语言性能条形图"""
        Visualizer._set_plt_style()
        
        if not results:
            print("Warning: Results dictionary is empty, skipping visualization")
            raise ValueError("Results dictionary is empty")

        # 数据准备
        data = []
        for lang, metrics in results.items():
            data.append({
                'Language': lang,
                'Precision': metrics.get('precision', 0),
                'Recall': metrics.get('recall', 0),
                'F1': metrics.get('f1', 0)
            })
        
        df = pd.DataFrame(data)
        
        plt.figure(figsize=(12, 6))
        x = np.arange(len(df['Language']))
        width = 0.25
        
        plt.bar(x - width, df['Precision'], width, label='Precision')
        plt.bar(x, df['Recall'], width, label='Recall')
        plt.bar(x + width, df['F1'], width, label='F1 Score')
        
        plt.xlabel('Language')
        plt.ylabel('Score')
        plt.title('Performance Across Languages')
        plt.xticks(x, df['Language'])
        plt.legend()
        
        # 添加数值标签
        for i, metric in enumerate(['Precision', 'Recall', 'F1']):
            for j, value in enumerate(df[metric]):
                plt.text(x[j] + (i-1)*width, value, 
                        f'{value:.3f}', 
                        ha='center', va='bottom')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_confusion_matrix(confusion_matrix, labels, output_path=None):
        """绘制混淆矩阵热力图"""
        Visualizer._set_plt_style()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_matrix, 
                    annot=True, 
                    fmt='d',
                    cmap='Blues',
                    xticklabels=labels,
                    yticklabels=labels)
        
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_entity_distribution(entity_counts, output_path=None):
        """绘制实体类型分布柱状图"""
        Visualizer._set_plt_style()
        
        plt.figure(figsize=(12, 6))
        
        # 转换数据格式
        entities = list(entity_counts.keys())
        counts = list(entity_counts.values())
        
        # 创建柱状图
        plt.bar(entities, counts)
        
        plt.title('Distribution of Entity Types')
        plt.xlabel('Entity Type')
        plt.ylabel('Count')
        
        # 旋转x轴标签以防重叠
        plt.xticks(rotation=45, ha='right')
        
        # 添加数值标签
        for i, count in enumerate(counts):
            plt.text(i, count, str(count),
                    ha='center', va='bottom')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_sequence_length_distribution(lengths, output_path=None):
        """绘制序列长度分布直方图"""
        Visualizer._set_plt_style()
        
        plt.figure(figsize=(10, 6))
        
        plt.hist(lengths, bins=50, edgecolor='black')
        plt.title('Distribution of Sequence Lengths')
        plt.xlabel('Sequence Length')
        plt.ylabel('Frequency')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()





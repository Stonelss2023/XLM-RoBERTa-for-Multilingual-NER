import os
import requests
import torch
from datasets import Dataset, DatasetDict
import pandas as pd
import numpy as np
from datasets import concatenate_datasets 

from DL11_4_0_config import Config
from DL11_4_1_data_loader_multilingual import DataLoader
from DL11_4_2_tokenizer_utils import TokenizerManager
from DL11_4_3_XLMR_model import NERModel
from DL11_4_4_NER_trainer import NERTrainer
from DL11_4_5_NER_evaluator import NEREvaluator
from DL11_4_6_NER_visualization import Visualizer

# 设置代理
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

# 测试连接
try:
    response = requests.get('https://huggingface.co')
    print("Connection successful!")
except:
    print("Connection failed! Please check whether the port is correct.")

def train_and_evaluate(
    model,
    tokenizer,
    training_datasets,
    eval_dataset,
    output_dir,
    training_mode,
    processed_datasets
):
    """训练并评估模型"""
    trainer = NERTrainer(
        model=model,
        tokenizer=tokenizer,
        output_dir=output_dir,
        train_dataset=training_datasets,
        eval_dataset=eval_dataset,
        training_mode=training_mode
    )
    
    evaluator = NEREvaluator(model.config.id2label)

    print(f"\nStarting {training_mode} training...")
    train_result = trainer.train()

    results = {}
    for lang in Config.LANGS:
        print(f"\nEvaluating {lang}...")
        if lang not in processed_datasets:
            print(f"Warning: No evaluation dataset for language {lang}")
            results[lang] = {'precision': 0, 'recall': 0, 'f1': 0}
            continue

        try:
            test_dataset = processed_datasets[lang]["test"]
            print("\nGenerating predictions...")
            
            # 修改预测逻辑
            raw_predictions = trainer.predict(test_dataset)
            if isinstance(raw_predictions, tuple):
                predictions = raw_predictions[0]
                labels = raw_predictions[1]
            else:
                predictions = raw_predictions.predictions
                labels = raw_predictions.label_ids
            
            # 处理维度问题
            if len(predictions.shape) > 2:
                predictions = predictions.argmax(axis=-1)
            if len(labels.shape) > 2:
                labels = labels.squeeze()
            
            # 确保预测和标签是2D数组
            predictions = predictions.reshape(-1)
            labels = labels.reshape(-1)
            
            # 计算指标
            detailed_metrics = evaluator.compute_metrics((predictions, labels))
            results[lang] = detailed_metrics
            print(f"Results for {lang}: {detailed_metrics}")
            
        except Exception as e:
            print(f"Error evaluating {lang}: {str(e)}")
            results[lang] = {'precision': 0, 'recall': 0, 'f1': 0}
            continue

    # 保存训练历史和性能图
    try:
        training_history = trainer.get_training_history()
        history_plot_path = os.path.join(output_dir, f"{training_mode}_training_history.png")
        Visualizer.plot_training_history(training_history, history_plot_path)

        performance_plot_path = os.path.join(output_dir, f"{training_mode}_language_performance.png")
        Visualizer.plot_language_performance(results, performance_plot_path)
    except Exception as e:
        print(f"Error in visualization: {str(e)}")

    return trainer, results

def main():
    # 设置环境
    Config.setup_environment()
    print("Environment setup completed.")

    # 初始化数据加载器和分词器管理器
    data_loader = DataLoader()
    tokenizer_manager = TokenizerManager()
    print("Initialized data loader and tokenizer manager.")

    # 加载数据集
    try:
        datasets = data_loader.load_datasets()
        print("Datasets loaded successfully.")
    except Exception as e:
        print(f"Error loading datasets: {str(e)}")
        return

    # 获取标签映射
    label_list = datasets[Config.LANGS[0]]["train"].features["ner_tags"].feature.names
    num_labels = len(label_list)
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}

    # 创建模型
    model = NERModel(num_labels, id2label, label2id)
    print("Model initialized.")

    # 处理数据集
    processed_datasets = {}
    for lang in Config.LANGS:
        if lang in datasets:
            processed_datasets[lang] = {}
            for split in datasets[lang]:
                processed_datasets[lang][split] = datasets[lang][split].map(
                    tokenizer_manager.tokenize_and_align_labels,
                    batched=True,
                    remove_columns=datasets[lang][split].column_names
                )

    # 对每种训练模式进行训练和评估
    for mode in Config.TRAINING_MODE:
        print(f"\nStarting {mode} training mode...")

        if mode == "transfer":
            train_dataset = processed_datasets["en"]["train"]
            eval_dataset = processed_datasets["en"]["validation"]
            output_dir = Config.TRANSFER_MODEL_DIR
        else:  # joint training
            joint_train = []
            joint_eval = []
            for lang in Config.LANGS:
                if lang in processed_datasets:
                    joint_train.append(processed_datasets[lang]["train"])
                    joint_eval.append(processed_datasets[lang]["validation"])
            train_dataset = concatenate_datasets(joint_train)
            eval_dataset = concatenate_datasets(joint_eval)
            output_dir = Config.JOINT_MODEL_DIR

        # 训练和评估
        trainer, results = train_and_evaluate(
            model=model.get_model(),
            tokenizer=tokenizer_manager.xlmr_tokenizer,
            training_datasets=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=output_dir,
            training_mode=mode,
            processed_datasets=processed_datasets
        )

        # 保存模型
        try:
            # 使用 model.save_pretrained 替代 trainer.save_model
            model.get_model().save_pretrained(output_dir)
            tokenizer_manager.xlmr_tokenizer.save_pretrained(output_dir)
            print(f"\n{mode.capitalize()} training completed. Model saved to {output_dir}")
        except Exception as e:
            print(f"Error saving model: {str(e)}")

        # 打印最终结果
        print("\nFinal Results:")
        for lang, result in results.items():
            print(f"{lang}: F1={result.get('f1', 0):.4f}")

if __name__ == "__main__":
    main()





from datasets import load_dataset, DatasetDict, concatenate_datasets
from collections import defaultdict, Counter
import pandas as pd
from DL11_4_0_config import Config

class DataLoader:
    def __init__(self):
        self.langs = Config.LANGS
        self.fracs = Config.LANG_FRACS
        self.cache_dir = "./dataset_cache"
    
    def load_datasets(self):
        """加载多语言数据集"""
        try:
            # 获取XTREME数据集中的PAN-X子集
            print("Loading PAN-X datasets...")
            return self.load_panx_datasets()
        except Exception as e:
            print(f"Error loading datasets: {str(e)}")
            raise

    def load_panx_datasets(self):
        """加载所有语言的PAN-X数据集"""
        panx_ch = defaultdict(DatasetDict)
        
        for lang, frac in zip(self.langs, self.fracs):
            try:
                print(f"Loading dataset for language: {lang}")
                ds = load_dataset(
                    "xtreme", 
                    f"PAN-X.{lang}",
                    cache_dir=self.cache_dir,
                    trust_remote_code=True
                )
                
                for split in ds:
                    panx_ch[lang][split] = (
                        ds[split]
                        .shuffle(seed=0)
                        .select(range(int(frac * ds[split].num_rows)))
                    )
            except Exception as e:
                print(f"Error loading dataset for language {lang}: {str(e)}")
                raise
                
        return panx_ch

    def encode_datasets(self, datasets, tokenizer):
        """对数据集进行编码"""
        encoded_datasets = {}
        
        def encode_batch(batch):
            # 确保 tokens 是列表格式
            if isinstance(batch["tokens"], list):
                texts = batch["tokens"]
            else:
                texts = batch["tokens"].tolist()
                
            # 处理每个文本
            processed_texts = []
            for text in texts:
                if isinstance(text, list):
                    processed_texts.append(" ".join(text))
                else:
                    processed_texts.append(text)
            
            # 进行分词
            encodings = tokenizer(
                processed_texts,
                truncation=True,
                padding=True,
                is_split_into_words=False,
                return_tensors=None
            )
            
            return encodings
        
        for lang, ds in datasets.items():
            try:
                encoded_ds = {}
                for split in ['train', 'validation', 'test']:
                    if split in ds:
                        encoded_ds[split] = ds[split].map(
                            encode_batch,
                            batched=True,
                            remove_columns=ds[split].column_names
                        )
                encoded_datasets[lang] = DatasetDict(encoded_ds)
            except Exception as e:
                print(f"Error encoding dataset for language {lang}: {str(e)}")
                raise
                
        return encoded_datasets

    def prepare_joint_dataset(self, encoded_datasets):
        """准备联合训练数据集
        
        Args:
            encoded_datasets: 编码后的多语言数据集字典，格式为:
                {lang: DatasetDict({split: Dataset})}
        
        Returns:
            DatasetDict: 包含合并后的训练、验证和测试集的数据集
        """
        try:
            print("Preparing joint training dataset...")
            joint_dataset = DatasetDict()
            
            for split in ['train', 'validation', 'test']:
                # 收集所有语言的相同split数据集
                datasets_to_concat = []
                for lang, ds in encoded_datasets.items():
                    if split in ds:
                        # 添加语言标识
                        current_ds = ds[split].add_column(
                            "language",
                            [lang] * len(ds[split])
                        )
                        datasets_to_concat.append(current_ds)
                
                if datasets_to_concat:
                    # 合并数据集并打乱
                    joint_dataset[split] = concatenate_datasets(
                        datasets_to_concat
                    ).shuffle(seed=42)
                    print(f"Created {split} set with {len(joint_dataset[split])} examples")
                
            return joint_dataset
            
        except Exception as e:
            print(f"Error preparing joint dataset: {str(e)}")
            raise
    
    def get_dataset_statistics(self, datasets):
        """获取数据集统计信息
        
        Args:
            datasets: 数据集字典
            
        Returns:
            dict: 包含每种语言每个分割的样本数量
        """
        stats = defaultdict(dict)
        for lang, ds in datasets.items():
            for split in ds.keys():
                stats[lang][split] = len(ds[split])
        return stats
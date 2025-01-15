✳ 从monolingual到multilingual如何转变？
1. 直接在Hugging Face Hub上调用对应语言的预训练模型？ → 低资源语言受限严重,没有pretrained model
2. Managing multiple monolingual simultaneously pretrained model could be challenging!

∴We use a class of multilingual transformers. Like BERT, multilingual tansformers are also pretrained by using masked language model tasks(MLM), but they are trained jointly on texts in over one hundred languages, enabling zero-shot cross-lingual transfer.This means that a model that is fine-tuned on one language can be applied to others without any further training!

✳ The target of the project is to realize Multilingual NER(named entity recognition)

Named Entity Rocognition has various downstream applications such as:
1. Gaining insights from company documents
2. Augment the quality of searching engines
3. Building a structured database from a corpus

✳ Zero-shot transfer = zero-shot learning → 现代大语言模型最吸引人的部分
1. Referring to the task of training a model on one set of labels and then evaluating it on a different set of labels.(如：英语NER数据训练→直接用于德语NER;新闻文本分类训练→直接用于医疗文本分类)
2. 在Transformer语境下的特殊含义:A language modle is evaluated on a downstream task that it wasn't even fine-tuned on(一般性语言理解与生成pretraining→直接zero-shot transfer到translation/summarization/classification等具体downstream tasks！)

# 1. The Dataset
(1) To load one of the PAN-X subsets in XTREME, we'll need to know which dataset configuration to pass the load_dataset() function. Whenever you're dealing with a dataset that has mutiple domains, you can use the "get_dataset_config_names()" function to find out which subsets are available:

(2) Then, we can narrow the search by just looking for the configurations that start with "PAN" using the method -- .startswith()

['PAN-X.af', 'PAN-X.ar', 'PAN-X.bg']

(3)得知subset形式以后,可以给load_dataset()函数中的argument"name"传入指定suffix,即可导入对应语言subset
如德语PAN数据指定导入：load_dataset("xtreme", name="PAN-X.de")

(4)Sample German(de), French(fr), Italian(it), English(en) corpora according to their proportion to make a realistic Swiss corpus. Create a Python defaultdict that stores the language code → key; a PAN-X corpus of type DataDict → value


```python
import pandas as pd
from datasets import get_dataset_config_names
from datasets import load_dataset    
from collections import defaultdict
from datasets import DatasetDict
from collections import Counter

xtreme_subsets = get_dataset_config_names("xtreme")
print(f"XTREME has {len(xtreme_subsets)} configurations")

panx_subsets = [s for s in xtreme_subsets if s.startswith("PAN")]
print(panx_subsets[:3])

load_dataset("xtreme", name="PAN-X.de") #de是德语


langs = ["de", "fr", "it", "en"]
fracs = [0.629, 0.229, 0.084, 0.059]
panx_ch = defaultdict(DatasetDict)

for lang, frac in zip(langs, fracs):
    ds = load_dataset("xtreme", name=f"PAN-X.{lang}")

    for split in ds:
        panx_ch[lang][split] = (
            ds[split]
            .shuffle(seed=0)
            .select(
                range(
                    int(frac * ds[split].num_rows)
                )
            )
        )

print(pd.DataFrame({lang: [panx_ch[lang]["train"].num_rows] for lang in langs},
            index=["Number of training examples"]))

element = panx_ch["de"]["train"][0]
for key, value in element.items():
    print(f"{key}: {value}")
    
for key, value in panx_ch["de"]["train"].features.items():
    print(f"{key}: {value}")

tags = panx_ch["de"]["train"].features["ner_tags"].feature
print(tags)
```

✳ 步骤：
3. 加载语言的完整数据集(PAN-X.{lang}是一个格式模板,其中{lang}是占位符)
    逐个替换过程通过字符串格式化方法实现：# f-strings方式（Python 3.6+）
4. 对每个数据分割(train/test/val)
(1) ds[split] 获取原始训练数据集
(2) .shuffle(seed=0)随机打乱数据,随机种子设定保证每次打乱顺序相同
(3) .select()选择部分数据;计算过程range(int(frac * ds[split].num_rows)(.num_row是数据集的一个属性用来获取总行数)
5. 打乱数据 → 计算需要保留的数据量 → 选择相应数量的数据 → 存储到相应位置

## XLM-RoBERTa 相比于 BERT的主要创新点

1. 多语言预训练,不依赖翻译语料
2. 训练目标仅保留 masked language modeling → 消除对平行语料的依赖
3. 模型规模扩大: 更大的词表(250k tokens) → 更强的多语言表征能力
4. 数据采样策略：按照真实语境比重加权采样 → 防止高资源语言主导训练,提升低资源语言表征
5. 此表构建：使用SentencePiece分词 → 支持更多语种分词 



# 2. A Closer Look at Tokenizer

在以往经验中tokenizer通常被看作一步operation,但实际上完整pipeline是被称为wordpiece的分词方法.先后经历"Normalization"→"Pretokenization"→"tokenizer model"→"postprocessing"四步。这种tokenization method主要面向英语等印欧系语言(依赖空哦哥哥和标点符号进行分词),但是对于"Chinese,Japanese,Korea"等于言而言就不方便

另一种后来居上的分词方法被称为sentencepiece tokenization. "采用Unicode等预定义方式将多语言文本全部转换为预定义编码"→"模型自己学习‘最佳子词组合与切分’"→'按照学习所得依照频率规律给出具体编码'

在跨语言任务以及最新的语言模型预处理选择中更倾向于sentencepiece方法,因为其语言无关性强,能欧冠适应多种语言场合。同时,对空格有特定编码可以保存空格信息,在处理非空格分词语言是表现更加(但是unicode加载对硬件性能要求高)

由于BERT等部分早期模型使用的是wordpiece分词,所以为了部分场景下的可兼容性仍会使用wordpiece.但总体来说sentencepiece分词已经超越wordpiece分词成为主流

SentencePiece的核心思想是数据驱动
1. 收集大量目标语言文本 → monolingual/multilingual(无需任何预定义规则)
2. 统计学习:使用Unigram算法 → 基于频率和互信息,自动学习最优切分,动态构建词表
3. 词表生成(设定词表大小):选择最优子词单元,自动生成编码映射

✳Unicode是预定义好的国际通用字符集编码 → 字符集序列,最小单元 → 只可组合不可拆分

【WordPiece流程】Normalization → preprocessing(基础词单元) → subtoken(训练习得) → 词表映射(训练习得)token IDs

【SentencePiece流程】Unicode编码(标准映射) → 子词切分(训练习得) → 词表映射(token IDs依据词频给)

✳ SentencePiece优势：无需语言相关规则 + 完美还原原始文本信息

# 3. Loading a custom model


```python
bert_model_name = "bert-base-cased"
xlmr_model_name = "xlm-roberta-base"
bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
xlmr_tokenizer = AutoTokenizer.from_pretrained(xlmr_model_name)

text = "Jack Sparrow loves New York!"
bert_tokens = bert_tokenizer(text).tokens()
xlmr_tokens = xlmr_tokenizer(text).tokens()
print(bert_tokens)
print(xlmr_tokens)

index2tag = {idx:tag for idx, tag in enumerate(tags.names)}
tag2index = {tag: idx for idx, tag in enumerate(tags.names)}

from transformers import (
    AutoConfig,
    AutoTokenizer,
    XLMRobertaForTokenClassification
)

xlmr_config = AutoConfig.from_pretrained(xlmr_model_name,
                                         num_labels=tags.num_classes,
                                         id2label=index2tag, label2id=tag2index)

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
xlmr_model = (XLMRobertaForTokenClassification.
              from_pretrained(xlmr_model_name, config=xlmr_config)
              .to(device))

input_ids = xlmr_tokenizer.encode(text, return_tensors="pt)
print(pd.DataFrame([xlmr_tokens, input_ids[0].numpy()],
                    index=["Token", "Inputs IDs"]))

outputs = xlmr_model(input_ids.to(device)).logits
predictions = torch.argmax(outputs, dim=-1)
print(f"Number of tokens in sequence: {len(xlmr_tokens)}")
print(f"Shape of outputs: {outputs.shape}")

preds = [tags.name[p] for p in predictions[0].cpu.numpy()]
print(pd.DataFrame([xlmr_tokens, preds], index=["Token", "Tags"]))


def tag_text(text, tags, model, tokenizer):
    tokens = tokenizer(texts).tokens()
    input_ids = xlmr_tokenizer(text, return_tensors="pt").inputs_ids.to(device)
    outputs = model(input_ids)[0]
    predictions = torch.argmax(outputs, dim=2)
    preds = [tags.names[p] for p in predictions[0].cpu().numpy()]
    return pd.DataFrame([tokens, preds], index=["Tokens", "Tags"])
```

1. ClassLabel——一个专门处理分类标签的类,此对象包含以下重要attribute
(1) .names: 所有标签名称的列表
(2) .num_classes: 标签总数
(3) .str2int: 标签名到数字的映射函数
(4) .int2str: 数字到标签的映射函数
∴ index2tag / tag2index实际上实在手动创建ClassLabel类似的映射功能,只是用更直接的字典形式存储
2. .features()是属性(property)用于获取数据集特征<tokens, ner_tags, lang> → 返回数据集所有特征的字典 / feature=是构造函数参数,用在定义Sequence时,指定序列二中单个元素的类型
3. AutoConfig.from_pretrained()会自动加载预训练模型的默认配置。这里我们通过在加载时覆盖num_labels,is2label,labels2id参数,从而在AutoConfig对象中保存customed的参数
4. AutoConfig.from_pretrained()只加载配置信息,不包含具体的矩阵权重。/ AutoModel.from_pretrained()加载模型的配置+权重信息(来源于预训练checkpoint) → ∴一般的流程是先用AutoConfig.from_pretrained()加载并覆盖部分配置信息,然后在AutoModel中传入覆盖后的config parameter(传入什么覆盖什么,未传入的一律保持默认值)
5. tokenizer处理输入文本时,即使只输入一个句子,也会返回一个batch格式的结果(2dim数组).0-dim是batch,1-dim是实际的token-ids(保持接口一致性 + 便于批处理<大多数深度学习框架下的操作都期望batch在第一维>)
6. output维度[batch_size, sequence_length, num_labels(类别及对应分数)] / .logits获取类别原始分数,如果不用则会输出模型完整对象,包含多个属性如attention_weights,hidden_states...
7. 封装进函数后的完整流程: Get tokens with special characters → Encode the sequence into IDs → take argmax to get most likely class per token → convert to DataFrame
8. GPU tensor不能直接准换numpy,所以要移到cpu上进行转换(DataFrame倾向于处理numpy)

# 4. Tokenizing for NER


```python
def tokenize_and_algin_labels(examples):
    """处理批量数据的主函数"""
    tokenized_inputs = xlmr_tokenizer(examples["tokens"],
                                    truncation=True,
                                    is_split_into_words=True)
    labels = []
    #获取ner_tags并确保是列表的列表 → batch样本处理时期待的数据结构
    ner_tags = examples["ner_tags"]
    if not isinstance(ner_tags[0], list): #检查第一个元素是否为列表
        ner_tags = [ner_tags]

    for idx, label_sequence in enumerate(ner_tags):
        #对于单个样本,batch_index应该是None
        word_ids = tokenized_inputs.word_ids() if len(ner_tags) == 1 else tokenized_inputs.word_ids(batch_index=idx)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None or word_idx==previous_word_idx:
                label_ids.append(-100)
            else:
                if isinstance(label_sequence, list) and word_idx < len(lables_sequence):
                    label_ids.append(label_sequence[word_idx])
                else:
                    label_ids.append(-100)
            previous_word_idx = word_idx

        labels.append(label_ids)

tokenized_inputs["labels"] = labels
return tokenized_inputs


encode_dataset(corpus):
    """对整个数据集进行编码"""
return corpus.map(tokenize_and_align_labels,
                  batched=True,
                  remove_columns=['langs', 'ner_tags', 'tokens'])
```

## 期待Nested List格式数据
列表的列表格式数据在NLP任务中非常常见,有以下主要场景:

1. 数据批量处理是的数据组织
2. 词级别标签序列的标注
3. 模型评估中"预测结果"与"真实标签"的对比
4. 分词处理得到子词分隔结果
5. 多层次化特征表示
6. 多头注意力权重 对应注意力计算

✳ 常见问题 及 解决方法

1. 长度不一致:需要padding对齐
2. 掩码处理：mask标记无效位置
3. 批量计算与数据类型统一：torch.tensor / .cpu().numpy
1. 批处理数据
batch_structure = {
    "输入数据": {
        "tokens": [
            ["我", "爱", "北京"],  # 第1个句子
            ["他", "在", "上海"]   # 第2个句子
        ],
        "features": [
            [1, 2, 3],  # 第1个句子的特征
            [4, 5, 6]   # 第2个句子的特征
        ]
    },
    "标签数据": {
        "ner_tags": [
            ["O", "O", "B-LOC"],  # 第1个句子的标签
            ["O", "O", "B-LOC"]   # 第2个句子的标签
        ]
    }
}

2. 序列标准评估
评估数据 = {
    "真实标签": [
        ["O", "B-PER", "I-PER"],  # 第1个样本
        ["O", "B-ORG", "I-ORG"]   # 第2个样本
    ],
    "预测标签": [
        ["O", "B-PER", "O"],      # 第1个样本的预测
        ["O", "B-ORG", "I-ORG"]   # 第2个样本的预测
    ]
}

3. 分词/子词处理
tokenization = {
    "原始输入": ["Hello World", "Deep Learning"],
    "分词结果": [
        ["Hello", "##llo", "World"],    # 第1句的子词
        ["Deep", "Learn", "##ing"]      # 第2句的子词
    ],
    "对应ID": [
        [101, 102, 103],  # 第1句的token_ids
        [104, 105, 106]   # 第2句的token_ids
    ]
}

4. 注意力机制
attention_data = {
    "注意力权重": [
        [[0.1, 0.2], [0.3, 0.4]],  # 第1个头的注意力矩阵
        [[0.5, 0.6], [0.7, 0.8]]   # 第2个头的注意力矩阵
    ]
}
## 5. Performance Measure


```python
import numpy as np
from seqeval.metrics import classification
def align_predictions(predictions, label_ids):
    preds = np.argmax(predictions, axis=2)
    batch_size, seq_len = preds.shape
    labels_list, preds_list = [], []

    for batch_idx in range(batch_size):
        example_labels, example_preds = [], []
        for seq_idx in range(seq_len):
            if label_ids[batch_idx, seq_idx] != -100:
                example_labels.append(index2tag[label_ids[batch_idx][seq_idx]])
                example_preds.append(index2tag[preds[batch_idx][seq_idx]])

            labels_list.appedn(example_labels)
            preds_list.append(example_preds)

    return preds_list, labels_list
    
```

## 期待的数据格式
此处align_predicitons函数专注于把模型输出的tensor格式转换成seqeval需要的list of lists格式,classification_report是评估函数,在得到正确格式的数据后使用
# 三个核心指标
指标 = {
    "Precision(精确率)": {
        "含义": "预测为正例中真实正例的比例",
        "公式": "真正例/(真正例+假正例)",
        "特点": "越高说明预测的正例越准确"
    },
    "Recall(召回率)": {
        "含义": "真实正例中被正确预测出的比例", 
        "公式": "真正例/(真正例+假负例)",
        "特点": "越高说明漏掉的正例越少"
    },
    "F1-score": {
        "含义": "Precision和Recall的调和平均数",
        "公式": "2 * (Precision * Recall)/(Precision + Recall)",
        "特点": "平衡了Precision和Recall"
    }
}# 输入 predictions 是三维tensor/array:
predictions = [          # 第1维：batch_size (批次大小)
    [                   # 第2维：seq_len (序列长度)
        [0.1, 0.8, 0.1],  # 第3维：num_labels (每个位置的标签概率分布)
        [0.7, 0.2, 0.1],
        ...
    ],
    [...],  # 第二个序列
    ...     # 更多序列
]
# shape: [batch_size, seq_len, num_labels]

# np.argmax(predictions, axis=2) 后变成二维:
preds = [              # 第1维：batch_size
    [1, 0, 2, ...],   # 第2维：seq_len (只保留最大概率的标签索引)
    [2, 1, 0, ...],
    ...
]
# shape: [batch_size, seq_len]

# 最终输出 preds_list 是二维列表:
preds_list = [                    # 第1维：batch中的序列
    ["B-PER", "I-PER", "O"],     # 第2维：单个序列中的标签
    ["B-ORG", "I-ORG", "O"],
    ...
]

三维 → 二维 → 二维
[批次,[序列,[概率]]] → [批次,[标签ID]] → [序列,[标签文本]]
# 6. Fine-Tuning XLM-RoBERTa
Now all the ingredients to fine-tune our model are ready! The first strategy will be to fine-tune our base model on the German subset of PAN-X and then evaluate its zero-shot cross-lingual performance on French, Italian and English. As usual, we'll use the Huggingface Transformers Trainer to handle our training loop, so first we need to define the training attributes using the Trainging Arguments class


```python
from transformers import TrainingArguments

num_epochs = 3
batch_size = 24
logging_steps = len(panx_de_encoded["train"]) // batch_size
model_name = f"{xlmr_model_name}-finetuned-panx-de"
# 训练参数配置
training_args = TrainingArguments(
    output_dir=model_name, log_level="error", num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size, evaluation_strategy="epoch",
    save_steps=1e6, weight_decay=0.01, disable_tqdm=False,
    logging_steps=loggin_steps, push_to_hub=True)

notebook_login()  


from seqeval.metrics import f1_score
def compute_metrics(eval_pred):
    y_pred, y_true = align_predictions(eval_pred.predictions,
                                       eval_pred.label_ids)
    return {"f1": f1_score(y_true, y_pred)}


from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification
data_collator = DataCollatorForTokenClassification(xlmr_tokenizer)

def model_init():
    return (XLMRobertaForTokenClassification
            .from_pretrained(xlmr_model_name, config=xlmr_config)
            .to(device))
    
```
"logging_steps": {
        "计算": "训练集大小 // batch_size",
        "含义": "日志记录间隔步数",
        "作用": "每处理多少批数据记录一次日志"}

TrainingArguments参数 = {
    "输出相关": {
        "output_dir": "保存模型的目录路径",
        "log_level": "error级别日志记录",
        "logging_steps": "日志记录频率",
        "push_to_hub": "是否推送到Hugging Face hub"
    },
    
    "训练相关": {
        "num_train_epochs": "训练轮数",
        "per_device_train_batch_size": "每个设备的训练批次大小",
        "per_device_eval_batch_size": "每个设备的评估批次大小",
        "evaluation_strategy": "epoch模式,每轮结束评估",
        "save_steps": "每1e6步保存一次模型",
        "weight_decay": "权重衰减率0.01用于防止过拟合"
    },
    
    "优化相关": {
        "disable_tqdm": "是否禁用进度条显示",
        "weight_decay": "L2正则化系数",
        "save_steps": "模型保存间隔"
    }
}
## 【补充笔记6--传统手动训练 vs Transformers训练方式】
# 传统手动训练循环设计
model = MyModel()
optimizer = torch.optim.Adam(model.parameters())
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, batch['labels'])
        loss.backward()
        optimizer.step()


# 使用Transformers Trainer简化训练
from transformers import Trainer 

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tain_dataset,
    eval_dataset=eval_dataset,
)
# 开始训练
trainer.train()
## 【补充笔记7--Data Collation】
问题背景：
    批处理要求：一个batch中的所有序列长度必须一致
    现实情况：不同文本/标签序列长度不一
    解决方案：使用padding补齐到"最大序列长度"
NER任务特殊性：
    token分类标签也是序列→也需要padding
    解决方案：用-100补全 
    原因：Pytorch损失函数会自动忽略-100标签
# 7. Cross-lingual Transfer


```python
from transformers import Trainer
trainer = Trainer(model_init=model_init, args=training_args,
                  data_collator=data_collator, compute_metrics=compute_metrics,
                  train_dataset=panx_de_encoded["train"],
                  eval_dataset=panx_de_encoded["validation"],
                  tokenizer=xlmr_tokenizer)

trainer.train() 


text_de = "Jeff Dean ist ein Informatiker bei Google in Kalifornien"
tag_text(text_de, tags, trainer.model, xlmr_tokenizer)


def get_f1_score(trainer, dataset):
    return trainer.predict(dataset).metrics["test_f1"]

f1_scores = defaultdict(dict)
f1_scores["de"]["de"] = get_f1_score(trainer, panx_de_encoded["test"])
print(f"F1-score of [de] model on [de] dataset: {f1_scores['de']['de']:.3f}")

text_fr = "Jeff Dean est informaticien chez Google en Californie"
tag_text(text_fr, tags, trainer.model, xlmr_tokenizer)


def evaluate_lang_performance(lang, trainer):
    panx_ds = encode_dataset(panx_ch[lang])
    return get_f1_score(trainer, panx_ds["test"])

f1_scores["de"]["fr"] = evaluate_lang_performance("fr", trainer)
print(f"F1-score of [de] model on [fr] dataset: {f1_scores['de']['fr']:.3f}")
f1_scores["de"]["it"] = evaluate_lang_performance("it", trainer)
print(f"F1-score of [de] model on [it] dataset: {f1_scores['de']['it']:.3f}")
f1_scores["de"]["en"] = evaluate_lang_performance("en", trainer)
print(f"F1-score of [de] model on [it] dataset: {f1_scores['de']['it']:.3f}")
```

# 8. The condition in which Zero-shot transfer make sense


```python
import matplotlib.pyplot as plt
# 在子集上训练的函数
def train_on_subset(dataset, num_samples):
    train_ds = dataset["train"].shuffle(seed=42).select(range(num_samples))
    valid_ds = dataset["validation"]
    test_ds = dataset["test"]
    training_args.logging_steps = len(train_ds) // batch_size
    
    trainer = Trainer(model_init=model_init, args=training_args,
                     data_collator=data_collator, compute_metrics=compute_metrics,
                     train_dataset=train_ds, eval_dataset=valid_ds, tokenizer=xlmr_tokenizer)
    trainer.train()
    
    if training_args.push_to_hub:
        trainer.push_to_hub(commit_message="Training completed!")
        
    f1_score = get_f1_score(trainer, test_ds)
    return pd.DataFrame.from_dict(
        {"num_samples": [len(train_ds)], "f1_score": [f1_score]})

# 编码法语数据集
panx_fr_encoded = encode_dataset(panx_ch["fr"])

# 测试小规模训练
training_args.push_to_hub = False
metrics_df = train_on_subset(panx_fr_encoded, 250)
print(metrics_df)

# 增加训练集大小进行实验
for num_samples in [500, 1000, 2000, 4000]:
    metrics_df = pd.concat(
        [metrics_df, train_on_subset(panx_fr_encoded, num_samples)],
        ignore_index=True) 
    
fig, ax = plt.subplots()
ax.axhline(f1_scores["de"]["fr"], ls="--", color="r")
metrics_df.set_index("num_samples").plot(ax=ax)

plt.legend(["Zero-shot from de", "Fine-tuned on fr"], loc="lower right")
plt.ylim((0, 1))
plt.xlabel("Number of Training Samples")
plt.ylabel("F1 Score")
plt.show()
```
low-resource场景的选择,可以量化的计算阈值

```python
# 评估每种语言的性能
for lang in langs:
    f1 = evaluate_lang_performance(lang, trainer)
    print(f"F1-score of [de-fr] model on [{lang}] dataset: {f1:.3f}")

# 在所有语言上分别微调
corpora = [panx_de_encoded]

# 排除德语的迭代
for lang in langs[1:]:
    training_args.output_dir = f"xlm-roberta-base-finetuned-panx-{lang}"
    ds_encoded = encode_dataset(panx_ch[lang])
    metrics = train_on_subset(ds_encoded, ds_encoded["train"].num_rows)
    f1_scores[lang][lang] = metrics["f1_score"][0]
    corpora.append(ds_encoded)

# 连接所有语言分割
corpora_encoded = concatenate_splits(corpora)

# 在多语言语料库上训练
training_args.logging_steps = len(corpora_encoded["train"]) // batch_size
training_args.output_dir = "xlm-roberta-base-finetuned-panx-all"

trainer = Trainer(model_init=model_init, args=training_args,
                 data_collator=data_collator, compute_metrics=compute_metrics,
                 tokenizer=xlmr_tokenizer, train_dataset=corpora_encoded["train"],
                 eval_dataset=corpora_encoded["validation"])

trainer.train()
''

# 生成最终评估数据
for idx, lang in enumerate(langs):
    f1_scores["all"][lang] = get_f1_score(trainer, corpora[idx]["test"])

scores_data = {"de": f1_scores["de"],
               "each": {lang: f1_scores[lang][lang] for lang in langs},
               "all": f1_scores["all"]}

f1_scores_df = pd.DataFrame(scores_data).T.round(3)
f1_scores_df.rename_axis(index="Fine-tune on", columns="Evaluated on", 
                        inplace=True)
print(f1_scores_df)
```

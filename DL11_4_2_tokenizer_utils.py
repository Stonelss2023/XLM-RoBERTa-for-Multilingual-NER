from transformers import AutoTokenizer
from DL11_4_0_config import Config

class TokenizerManager:
    def __init__(self):
        self.xlmr_tokenizer = AutoTokenizer.from_pretrained(Config.XLMR_MODEL_NAME)
        
    def tokenize_and_align_labels(self, examples):
        """处理批量数据的标记对齐"""
        tokenized_inputs = self.xlmr_tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            padding="max_length",
            max_length=128,
            return_tensors=None
        )
        
        labels = []
        for idx, label_sequence in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=idx)
            previous_word_idx = None
            label_ids = []
            
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label_sequence[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
                
            labels.append(label_ids)
        
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def encode_dataset(self, corpus):
        """对整个数据集进行编码"""
        return corpus.map(
            self.tokenize_and_align_labels,
            batched=True,
            remove_columns=['langs', 'ner_tags', 'tokens']
        )

    def tag_text(self, text, tags, model, return_tokens=False):
        """标注单个文本"""
        tokens = self.xlmr_tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True,
            padding="max_length",
            max_length=128
        ).to(Config.DEVICE)
        
        outputs = model(**tokens)
        predictions = outputs.logits.argmax(dim=-1)
        
        pred_tags = [
            tags[p.item()] for p in predictions[0]
            if tags[p.item()] != "O"
        ]
        
        if return_tokens:
            return tokens, pred_tags
        return pred_tags
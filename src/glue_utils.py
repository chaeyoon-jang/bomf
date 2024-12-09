import torch
from transformers import RobertaModel
import evaluate
from datasets import load_dataset #, load_metric

task_text_field_map = {
            'sst2': ['sentence'],
            'mrpc': ['sentence1', 'sentence2'],
            'qqp': ['question1', 'question2'],
            'mnli': ['premise', 'hypothesis'],
            'qnli': ['question', 'sentence'],
            'rte': ['sentence1', 'sentence2'],
        }

glue_task_num_labels = {
            'sst2': 2,
            'mrpc': 2,
            'qqp': 2,
            'mnli': 3,
            'qnli': 2,
            'rte': 2,
        }

loader_columns = [
            'input_ids',
            'label',
            'attention_mask'
        ]


def make_glue_data(task, tokenizer):
    
    text_fields = task_text_field_map[task]
    num_labels = glue_task_num_labels[task]
    
    dataframe = load_dataset("glue", task)
    
    def convert_to_features(example_batch, indices=None):
        
        if len(text_fields) > 1:
            texts_or_text_pairs = list(zip(example_batch[text_fields[0]], 
                                               example_batch[text_fields[1]]))
        else:
            texts_or_text_pairs = example_batch[text_fields[0]]
                
        features = tokenizer.batch_encode_plus(
            texts_or_text_pairs,
            max_length=512,
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            return_token_type_ids=True
        ) 
            
        features['label'] = example_batch['label']
        return features
    
    for split in dataframe.keys():
        dataframe[split] = dataframe[split].map(
            convert_to_features,
            batched=True,
            )
        columns = [c for c in dataframe[split].column_names if c in loader_columns]
        dataframe[split].set_format(type="torch", columns=columns)
    
    eval_splits = [x for x in dataframe.keys() if 'validation' in x]
    eval_data = [dataframe[x] for x in eval_splits]
    
    return dataframe['train'], eval_data[0]


class metrics:
    def __init__(self, task_flag):
        #self.metric_fn = load_metric('glue', task_flag)
        self.metric_fn = evaluate.load('glue', task_flag)
        self.task_flag = task_flag

    def calculate(self, logits, targets):
        result = None 
        
        if self.task_flag != 'stsb':
            _, preds = torch.max(logits, dim=-1)
            result = preds.eq(targets).sum().item()
            result = (result / preds.size(0)) * 100
            metric = self.metric_fn.compute(predictions=preds, references=targets)
        else:
            metric = self.metric_fn.compute(predictions=logits, references=targets)
        
        if self.task_flag == 'cola':
            metric = metric["matthews_correlation"] * 100
        elif self.task_flag == 'stsb':
            metric = metric["pearson"] * 100
            result = metric
        elif self.task_flag in ['mrpc', 'qqp']:
            metric = metric["f1"] * 100
        else:
            metric = result

        return result, metric
    

class RobertaGLUE(torch.nn.Module):
    def __init__(self, config):
        super(RobertaGLUE, self).__init__()
        self.roberta = RobertaModel.from_pretrained(config.model_type,
                                                    add_pooling_layer=False,
                                                    ignore_mismatched_sizes=True)
        
        for layer in self.roberta.encoder.layer[:config.freeze_num]:
            for param in layer.parameters():
                param.requires_grad = False
                
        self.dense = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(config.classifier_dropout)
        self.out = torch.nn.Linear(config.hidden_size, glue_task_num_labels[config.task])
    
    def forward(self, input_ids, attention_mask):
        
        outputs = self.roberta(input_ids=input_ids, 
                               attention_mask=attention_mask)

        sequence_output = outputs[0]
        logits = sequence_output[:, 0, :]  
        logits = self.dropout(logits)
        logits = self.dense(logits)
        logits = torch.tanh(logits)
        logits = self.dropout(logits)  
        logits = self.out(logits)

        return logits
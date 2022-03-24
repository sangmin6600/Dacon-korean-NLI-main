import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


import argparse
import random

from sklearn.metrics import accuracy_score

import torch

from transformers import BertTokenizerFast
from transformers import BertForSequenceClassification, AlbertForSequenceClassification
from transformers import Trainer
from transformers import TrainingArguments

from bert_dataset import TextClassificationCollator
from bert_dataset import TextClassificationDataset
from utils import read_text

from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer, AutoModel


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', type = str)
    p.add_argument('--train_fn', default = './data/data_aug.tsv')
    
    # Recommended model list:
    # - kykim/bert-kor-base
    # - kykim/albert-kor-base
    # - beomi/kcbert-base
    # - beomi/kcbert-large
    p.add_argument('--pretrained_model_name', type=str, default='klue/roberta-large')
    p.add_argument('--use_albert', action='store_true')

    p.add_argument('--valid_ratio', type=float, default=.2)
    p.add_argument('--batch_size_per_device', type=int, default=912)
    
    
    p.add_argument('--lr', type=float, default = 5e-5)
    p.add_argument('--n_epochs', type=int, default=5)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--warmup_ratio', type=float, default=.2)

    p.add_argument('--beta1', type=float, default=0.9)
    p.add_argument('--beta2', type=float, default=0.999)
    p.add_argument('--epsilon', type=float, default=1e-8)
    

    p.add_argument('--max_length', type=int, default=200)
    p.add_argument('--amp', type=bool, default=True)
    
    p.add_argument('--drop_out_p', type=float, default = 0.1)
    p.add_argument('--attention_drop_out_p', type=float, default = 0.1)
    p.add_argument('--out_dir', type = str, default = './models')
    
    
#     p.add_argument('--gpu_id', type = str, default = 0)
    
    

    config = p.parse_args()

    return config


def get_datasets(fn, valid_ratio=.2):
     # Get list of labels and list of texts.
    premises, hypothesises, labels = read_text(fn)

    # Generate label to index map.
    unique_labels = list(set(labels))
    label_to_index = {}
    index_to_label = {}
    for i, label in enumerate(unique_labels):
        label_to_index[label] = i
        index_to_label[i] = label

    # Convert label text to integer value.
    labels = list(map(label_to_index.get, labels))

    # Shuffle before split into train and validation set.
    shuffled = list(zip(premises, hypothesises, labels))
    random.shuffle(shuffled)
    premise = [e[0] for e in shuffled]
    hypothesis = [e[1] for e in shuffled]
    label = [e[2] for e in shuffled]
    idx = int(len(premises) * (1 - valid_ratio))

    train_dataset = TextClassificationDataset(premise[:idx], hypothesis[:idx], label[:idx])
    valid_dataset = TextClassificationDataset(premise[idx:], hypothesis[idx:], label[idx:])

    return train_dataset, valid_dataset, index_to_label


def main(config):

    # Get pretrained tokenizer.
#     tokenizer = BertTokenizerFast.from_pretrained(config.pretrained_model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name)

    # Get datasets and index to label map.
    train_dataset, valid_dataset, index_to_label = get_datasets(
        config.train_fn,
        valid_ratio=config.valid_ratio
    )

    print(
        '|train| =', len(train_dataset),
        '|valid| =', len(valid_dataset),
    )

    total_batch_size = config.batch_size_per_device * torch.cuda.device_count()
    n_total_iterations = int(len(train_dataset) / total_batch_size * config.n_epochs)
    n_warmup_steps = int(n_total_iterations * config.warmup_ratio)
    print(
        '#total_iters =', n_total_iterations,
        '#warmup_iters =', n_warmup_steps,
    )

    # Get pretrained model with specified softmax layer.
    config_ = AutoConfig.from_pretrained(config.pretrained_model_name)
    config_.num_labels = 3
    config_.hidden_dropout_prob = config.drop_out_p
    config_.attention_probs_dropout_prob = config.attention_drop_out_p
    
    model = AutoModelForSequenceClassification.from_pretrained(config.pretrained_model_name, config = config_)

    training_args = TrainingArguments(
        output_dir=config.out_dir,
        num_train_epochs=config.n_epochs,
        per_device_train_batch_size=config.batch_size_per_device,
        per_device_eval_batch_size=config.batch_size_per_device,
        warmup_steps=n_warmup_steps,
        weight_decay=config.weight_decay,
        fp16=config.amp,
        learning_rate = config.lr,
        evaluation_strategy='epoch',
#         eval_steps = 500,

        save_strategy = 'epoch',
        
        logging_steps=n_total_iterations // config.max_length,
        save_steps=n_total_iterations // config.n_epochs,
        load_best_model_at_end=True,
        
        adam_beta1 = config.beta1,
        adam_beta2 = config.beta2,
        adam_epsilon = config.epsilon,
    )

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)

        return {
            'accuracy': accuracy_score(labels, preds)
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=TextClassificationCollator(tokenizer,
                                       config.max_length,
                                       with_text=False),
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()

    torch.save({
        'bert': trainer.model.state_dict(),
        'config': config,
        'classes': index_to_label,
        'tokenizer': tokenizer,
    }, f'./pth_files/{config.model_fn}_{config.drop_out_p}d_{1 - config.valid_ratio}p_{config.n_epochs}e_.pth')
#     model.save_pretrained('./models/' +str(config.model_fn)+str(config.n_epochs) )

if __name__ == '__main__':
    config = define_argparser()
    os.makedirs("./pth_files/klue/", exist_ok=True)

    for pt, max_len, epoch, drop in zip(['klue/roberta-large', 'klue/roberta-large', 'klue/roberta-large', 'klue/roberta-large',
                                   'klue/roberta-large', 'klue/roberta-large', 'klue/roberta-large', 'klue/roberta-large'],
                                  [65, 70, 70, 70, 70, 70, 70, 70, 70], #max_len
                                  [3, 3, 3, 4,  5,  5, 6, 6, 7],  #epochs
                                  [0.05, 0.07, 0.1, 0.12, 0.15, 0.17, 0.20, 0.22,0.25] #drop_out
                                 ):
        print(f'load - {pt}, max_len : {max_len}, epochs : {epoch}, drop_out : {drop}')
        config.n_epochs = epoch
        config.drop_out_p = drop
        config.attention_drop_out_p = drop
        config.model_fn = pt
        config.max_length = max_len
        main(config)

#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.
from sklearn.metrics import f1_score, accuracy_score
from dataHelper import get_dataset
import logging
import sys
from dataclasses import dataclass, field
from typing import Optional
import os
import datasets
import numpy as np
import transformers
from adapters.adapters import load_adapter_model
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    default_data_collator,
    set_seed,
)
from peft import LoraConfig, TaskType, get_peft_model


#from transformers.integrations import WandbCallback
logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )

    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_dir: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )

    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )

    peft: Optional[str] = field(
        default=None,
        metadata={"help": "decide whether to use Lora or Adapter to do PEFT"}
    )


# @dataclass
# class TrainingArguments:
#     per_device_train_batch_size: int = field(
#         default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
#     )
#     learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
#     seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of training."})
#     output_dir: str = field(
#         metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
#     )
#     num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})


def main():
    
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_PROJECT"] = f"{data_args.dataset_name} \
          {model_args.model_name_or_path.split('/')[-1]} {training_args.seed}"
    # os.environ["WANDB_LOG_MODEL"] = "checkpoint" 
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
   
    logger.info(f"Training/evaluation parameters {training_args}")

    set_seed(training_args.seed)


    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    raw_datasets = get_dataset(data_args.dataset_name, tokenizer.sep_token)
    
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, num_labels=len(raw_datasets['train'].unique('label')))
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        config=config )
    
    if model_args.peft == 'lora':
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=8,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)
    elif model_args.peft == 'adapter':
        model = load_adapter_model(model)

    else:
        raise NotImplementedError
        
    #Some models have set the order of the labels to use, so let's make sure we do use it.
    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    
    def tokenize(examples):
        if data_args.pad_to_max_length:
            return tokenizer(examples['text'], padding='max_length', max_length=max_seq_length, truncation=True)
        else:
            return tokenize(examples['text'])
    tokenized_datasets = raw_datasets.map(
        tokenize,
        batched=True,
        desc=f"Running tokenizer on every text in dataset"
    )
   
    
    train_dataset = tokenized_datasets["train"]
    test_dataset = tokenized_datasets["test"]

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        labels = p.label_ids
        
        preds= np.argmax(preds, axis=1)

        micro_f1 = f1_score(labels, preds, average='micro')
        macro_f1 = f1_score(labels, preds, average='macro')
        accuracy = accuracy_score(labels, preds)

        return {
            'micro_f1': micro_f1,
            'macro_f1': macro_f1,
            'accuracy': accuracy,
        }

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collactor = default_data_collator
    elif training_args.fp16:
        data_collactor = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collactor
    )

    train_result = trainer.train()
    metrics = train_result.metrics

    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate(eval_dataset=test_dataset)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics('eval', metrics)



if __name__ == "__main__":
    main()
    
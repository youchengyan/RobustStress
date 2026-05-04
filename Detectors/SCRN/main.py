from utils.utils import set_logger, path_checker, metrics_fn, compute_metrics

import torch
import numpy as np
import random
import pickle
import datetime
import json

from transformers import (AutoConfig, AutoModelForSequenceClassification, Trainer, HfArgumentParser, set_seed,
                          AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForCausalLM)

from arguments import ModelArguments, DataTrainingArguments, TrainingArguments
from datasets import load_dataset, DatasetDict, Dataset
from utils.scrn_model import SCRNModel, SCRNTrainer
from utils.utils import mask_tokens

import os

os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB__SERVICE_WAIT"] = "300"


class CustomDataCollatorForSeqCLS(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        return features


def metrics_fn(outputs):
    y_true = outputs.label_ids
    y_pred = outputs.predictions.argmax(-1)
    y_score = torch.tensor(outputs.predictions).softmax(-1).numpy()[:, 1]
    return compute_metrics(y_true, y_pred, y_score)


# python main.py --do_predict True --save_total_limit 5 --learning_rate 1e-4 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --num_train_epochs 2.0 --logging_steps 50 --gradient_accumulation_steps 1 --data_files /data5/storage4/wudauser05/yyc/RobustStressBench/domain_specific_model_specific/xsum  --output_dir ./data_out/scrn --test_files /data5/storage4/wudauser05/yyc/RobustStressBench/domain_specific_model_specific/xsum/test.json


# python main.py --do_predict True --save_total_limit 5 --learning_rate 1e-4 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --num_train_epochs 2.0 --logging_steps 50 --gradient_accumulation_steps 1 --data_files /data5/storage4/wudauser05/yyc/RobustStressBench/domain_specific_model_specific/xsum  --output_dir ./data_out/scrn --test_files /data5/storage4/wudauser05/yyc/RobustStressBench/domain_specific_model_specific/xsum/SynonymReplacement/filter/test_perturb_5.json

def main():
    # Get arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_abbr = "scrn"
    dataset_abbr = data_args.data_files.split('/')[-1]
    training_args.output_dir = training_args.output_dir + '_' + dataset_abbr

    basename = os.path.basename(data_args.test_files)  # → "test.json"
    file_name_without_ext = os.path.splitext(basename)[0]
    training_args.output_dir = os.path.join(
        training_args.output_dir,
        file_name_without_ext
    )

    os.makedirs(training_args.output_dir, exist_ok=True)

    # Set seed
    set_seed(2026)

    def load_json(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    base = data_args.data_files
    raw_dataset = DatasetDict({
        "train": Dataset.from_list(load_json(f"{base}/train.json")),
        "test": Dataset.from_list(load_json(data_args.test_files)),
    })

    raw_dataset = raw_dataset.rename_column("label", "labels")

    # Load model
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=data_args.max_seq_length,
        padding_side="right",
        use_fast=False,
    )
    model = SCRNModel(model_args.model_name_or_path, config)


    def preprocess_function_for_ranmask(examples):
        examples["text"] = mask_tokens(examples["text"], mask_token=tokenizer.mask_token)
        inputs = tokenizer(examples["text"], truncation=True)
        model_inputs = inputs
        return model_inputs

    def preprocess_function_for_seq_cls(examples):
        inputs = tokenizer(examples["text"], truncation=True)
        model_inputs = inputs
        return model_inputs

    if model_abbr == 'ranmask':
        train_data_preprocess_fn = preprocess_function_for_ranmask
        infer_data_preprocess_fn = preprocess_function_for_seq_cls
    else:
        train_data_preprocess_fn = preprocess_function_for_seq_cls
        infer_data_preprocess_fn = preprocess_function_for_seq_cls

    # Preprocess dataset
    train_dataset, test_dataset = raw_dataset["train"], raw_dataset["test"]

    with training_args.main_process_first(desc="train dataset map pre-processing"):
        train_dataset = train_dataset.map(
            train_data_preprocess_fn,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )
        test_dataset = test_dataset.map(
            infer_data_preprocess_fn,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on test dataset",
        )

    data_collator = CustomDataCollatorForSeqCLS(tokenizer, model=model,
                                                pad_to_multiple_of=8 if training_args.fp16 else None, )

    # Set trainer
    trainer = SCRNTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        eval_dataset=test_dataset,
        compute_metrics=metrics_fn,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train()
        # trainer.save_state()
        trainer.save_model()

    # Predict
    if training_args.do_predict:
        config = AutoConfig.from_pretrained(training_args.output_dir)
        model = SCRNModel(model_args.model_name_or_path, config=config)
        model.load_state_dict(torch.load(os.path.join(training_args.output_dir, 'pytorch_model.bin')))
        trainer = SCRNTrainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            data_collator=data_collator,
            eval_dataset=test_dataset,
            compute_metrics=metrics_fn,
        )
        predict_results = trainer.evaluate()
        trainer.save_metrics(file_name_without_ext+"_SCRN", predict_results)


if __name__ == "__main__":
    main()

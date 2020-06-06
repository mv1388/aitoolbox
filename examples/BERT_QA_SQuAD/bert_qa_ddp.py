import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    BertConfig,
    BertTokenizer,
)

from .lib.bert_model import BertQAModel
from .lib.data_prep import load_and_cache_examples
from .lib.transformers_qa_bert_eval import DDPBERTQAResultPackage

from aitoolbox.torchtrain.train_loop import TrainLoopCheckpointEndSave
import aitoolbox.torchtrain.tl_components.pred_collate_fns as collate_fns
from aitoolbox.torchtrain.callbacks.train_schedule import LambdaLRScheduler
from aitoolbox.torchtrain.callbacks.performance_eval import (
    ModelPerformanceEvaluation, ModelPerformancePrintReport, ModelTrainHistoryPlot, ModelTrainHistoryFileWriter)
from aitoolbox.torchtrain.callbacks.basic import EarlyStopping


def lr_lambda(current_step):
    warmup_steps = 0
    t_total = 55580.0
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return max(0.0, float(t_total - current_step) / float(max(1, t_total - warmup_steps)))


if __name__ == '__main__':
    dir_prefix = os.path.expanduser('~/project/')
    project_root = '~/project/model_results'

    data_dir = os.path.join(dir_prefix, 'data/SQuAD2')
    predict_file = f'{data_dir}/dev-v2.0.json'
    train_file = f'{data_dir}/train-v2.0.json'

    model_name_or_path = 'bert-base-cased'
    cache_dir = os.path.join(dir_prefix, 'data/bert_pretrained')
    do_lower_case = True

    max_seq_length = 384
    max_query_length = 64
    doc_stride = 128

    num_train_epochs = 5.

    train_batch_size = 12
    eval_batch_size = 8

    gradient_accumulation_steps = 1
    weight_decay = 0.
    learning_rate = 3e-5
    adam_epsilon = 1e-8
    warmup_steps = 0

    config_class, tokenizer_class = BertConfig, BertTokenizer

    config = config_class.from_pretrained(
        model_name_or_path, cache_dir=cache_dir
    )
    tokenizer = tokenizer_class.from_pretrained(
        model_name_or_path,
        do_lower_case=do_lower_case,
        cache_dir=cache_dir
    )
    model = BertQAModel(config, model_name_or_path, cache_dir)

    train_dataset = \
        load_and_cache_examples(tokenizer,
                                data_dir=data_dir, predict_file=predict_file, train_file=train_file,
                                model_name_or_path=model_name_or_path,
                                max_seq_length=max_seq_length, max_query_length=max_query_length, doc_stride=doc_stride,
                                overwrite_cache=True, evaluate=False, output_examples=False)
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

    test_dataset, test_examples, test_features = \
        load_and_cache_examples(tokenizer,
                                data_dir=data_dir, predict_file=predict_file, train_file=train_file,
                                model_name_or_path=model_name_or_path,
                                max_seq_length=max_seq_length, max_query_length=max_query_length, doc_stride=doc_stride,
                                overwrite_cache=True, evaluate=True, output_examples=True)
    test_dataloader = DataLoader(test_dataset, batch_size=eval_batch_size)

    t_total = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)

    callbacks = [LambdaLRScheduler(lr_lambda, last_epoch=-1),
                 ModelPerformanceEvaluation(DDPBERTQAResultPackage(test_examples, test_features, project_root, tokenizer), {}),
                 ModelPerformancePrintReport(['val_exact', 'val_f1']),
                 ModelTrainHistoryPlot(),
                 ModelTrainHistoryFileWriter(),
                 EarlyStopping(monitor='val_f1', patience=2)]

    print('START TRAINING LOOP')
    TrainLoopCheckpointEndSave(model,
                               train_dataloader, test_dataloader, None,
                               optimizer, nn.NLLLoss(),
                               project_name='GPU_eval_QA_SQuAD2', experiment_name='BERT_QA_DDP',
                               local_model_result_folder_path=project_root,
                               hyperparams={},
                               val_result_package=DDPBERTQAResultPackage(test_examples, test_features, project_root, tokenizer),
                               test_result_package=None,
                               source_dirs=['~/project/experiments/BERT_squad'],
                               rm_subopt_local_models=True, num_best_checkpoints_kept=3,
                               collate_batch_pred_fn=collate_fns.append_predictions,
                               pred_transform_fn=collate_fns.torch_cat_transf,
                               gpu_mode='ddp',
                               use_amp=False)\
        .fit(num_epochs=int(num_train_epochs), callbacks=callbacks,
             ddp_model_args={'find_unused_parameters': True})

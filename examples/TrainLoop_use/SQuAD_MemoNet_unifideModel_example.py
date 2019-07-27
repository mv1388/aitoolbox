from torch.utils.data import DataLoader
from torch import optim
import torch
import torch.nn as nn

from AIToolbox.nlp.dataset.SQuAD2.SQuAD2DataReader import SQuAD2ConcatContextDatasetReader
from AIToolbox.torchtrain.data.dataset import BasicDataset as SQuAD2Dataset
from AIToolbox.nlp.dataset.torch_collate_fns import qa_concat_ctx_span_collate_fn

from AIToolbox.torchtrain.train_loop import TrainLoopModelCheckpointEndSave
from AIToolbox.torchtrain.callbacks.performance_eval_callbacks import ModelPerformanceEvaluation, \
    ModelPerformancePrintReport, ModelTrainHistoryPlot
from AIToolbox.torchtrain.callbacks.train_schedule_callbacks import ReduceLROnPlateauScheduler
from AIToolbox.nlp.experiment_evaluation.NLP_result_package import QuestionAnswerResultPackage

from AIToolbox.nlp.models.torch.unified_qa_model import UnifiedQABasicRNN


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


project_folder_prefix = '~/PycharmProjects/RNN_QANet'
# project_folder_prefix = '~/project'


reader_train = SQuAD2ConcatContextDatasetReader(f'{project_folder_prefix}/data/SQuAD2/train-v2.0.json',
                                                is_train=True, dev_mode_size=2)
reader_dev = SQuAD2ConcatContextDatasetReader(f'{project_folder_prefix}/data/SQuAD2/dev-v2.0.json',
                                              is_train=False, dev_mode_size=2)
reader_test = SQuAD2ConcatContextDatasetReader(f'{project_folder_prefix}/data/SQuAD2/dev-v2.0.json',
                                               is_train=False, dev_mode_size=2)
data_train, vocab = reader_train.read()
data_dev, _ = reader_dev.read()
data_test, _ = reader_test.read()

data_train2idx = [(torch.Tensor(vocab.convert_sent2idx_sent(paragraph_tokens)),
                   torch.Tensor(vocab.convert_sent2idx_sent(question_tokens)),
                   span_tuple)
                  for paragraph_tokens, question_tokens, span_tuple, _ in data_train]

data_dev2idx = [(torch.Tensor(vocab.convert_sent2idx_sent(paragraph_tokens)),
                 torch.Tensor(vocab.convert_sent2idx_sent(question_tokens)),
                 span_tuple)
                for paragraph_tokens, question_tokens, span_tuple, _ in data_dev]

data_test2idx = [(torch.Tensor(vocab.convert_sent2idx_sent(paragraph_tokens)),
                 torch.Tensor(vocab.convert_sent2idx_sent(question_tokens)),
                 span_tuple)
                 for paragraph_tokens, question_tokens, span_tuple, _ in data_test]

train_ds = SQuAD2Dataset(data_train2idx)
dev_ds = SQuAD2Dataset(data_dev2idx)
test_ds = SQuAD2Dataset(data_test2idx)
train_loader = DataLoader(train_ds, batch_size=100, collate_fn=qa_concat_ctx_span_collate_fn)
dev_loader = DataLoader(dev_ds, batch_size=100, collate_fn=qa_concat_ctx_span_collate_fn)
test_loader = DataLoader(test_ds, batch_size=100, collate_fn=qa_concat_ctx_span_collate_fn)

output_size = max([max([len(el[0]) for el in data_train]), max([len(el2[0]) for el2 in data_dev])])


model = UnifiedQABasicRNN(hidden_size=50,
                          output_size=output_size,
                          embedding_dim=50, vocab_size=vocab.num_words,
                          ctx_n_layers=1, qus_n_layers=1, dropout=0.2)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

used_args = {'batch_size': 100, 'hidden_size': 50, 'ctx_n_layers': 1, 'qus_n_layers': 1, 'dropout': 0.2,
             'dev_mode_size': 2, 'lr': 0.001, 'num_epoch': 2}


qa_result_pkg_cp = QuestionAnswerResultPackage([paragraph_tokens for paragraph_tokens, _, _, _ in data_dev],
                                               target_actual_text=[paragraph_text for _, _, _, paragraph_text in data_dev],
                                               output_text_dir='tempData_checkpoint_callback'
                                               )

callbacks = [ModelPerformanceEvaluation(qa_result_pkg_cp, used_args,
                                                on_each_epoch=True,
                                                on_train_data=False, on_val_data=True),
             ModelPerformancePrintReport(['val_ROGUE'],
                                                 on_each_epoch=False, list_tracked_metrics=True),
             ReduceLROnPlateauScheduler(threshold=0.1, patience=2, verbose=True),
             ModelTrainHistoryPlot(epoch_end=True)
             ]


print('Starting train loop')

# TrainLoop(model,
#           train_loader, dev_loader,
#           QASpanSQuADModelFeedDefinition(), optimizer, criterion)(num_epoch=30, callbacks=callbacks)


# qa_result_pkg = QuestionAnswerResultPackage([paragraph_tokens for paragraph_tokens, _, _ in data_dev],
#                                             '~/project/model_results/tempData')

# TrainLoopModelCheckpointEndSave(model, train_loader, dev_loader, QASpanSQuADModelFeedDefinition(), optimizer, criterion,
#                                 project_name='fullQAModelRunTest',
#                                 experiment_name='MemoryNetPytorchTest',
#                                 local_model_result_folder_path='~/project/model_results',
#                                 hyperparams={},
#                                 result_package=qa_result_pkg)(num_epoch=3, callbacks=callbacks)


qa_result_pkg_final = QuestionAnswerResultPackage([paragraph_tokens for paragraph_tokens, _, _, _ in data_dev],
                                                  target_actual_text=[paragraph_text for _, _, _, paragraph_text in data_dev],
                                                  output_text_dir='tempData_final_val'
                                                  )

qa_res_test_pkg_fin = QuestionAnswerResultPackage([paragraph_tokens for paragraph_tokens, _, _, _ in data_test],
                                                  target_actual_text=[paragraph_text for _, _, _, paragraph_text in data_test],
                                                  output_text_dir='tempData_final_test'
                                                  )

TrainLoopModelCheckpointEndSave(model, train_loader, dev_loader, test_loader, optimizer, criterion,
                                project_name='unifiedModelExample',
                                experiment_name='MemoryNetPytorchTest',
                                local_model_result_folder_path=f'{project_folder_prefix}/model_results',
                                hyperparams=used_args,
                                val_result_package=qa_result_pkg_final,
                                test_result_package=qa_res_test_pkg_fin)\
    (num_epoch=3, callbacks=callbacks)


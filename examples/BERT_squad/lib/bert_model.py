import torch
from transformers import BertForQuestionAnswering

from aitoolbox import TTModel


class BertQAModel(TTModel):
    def __init__(self, config, model_name_or_path, cache_dir):
        super().__init__()
        self.bert_model = BertForQuestionAnswering.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=config,
            cache_dir=cache_dir,
        )

    def forward(self, **batch_data):
        return self.bert_model(**batch_data)

    def get_loss(self, batch_data, criterion, device):
        batch = tuple(t.to(device) for t in batch_data)
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
            "start_positions": batch[3],
            "end_positions": batch[4],
        }

        outputs = self(**inputs)
        loss = outputs[0]
        return loss

    def get_predictions(self, batch_data, device):
        batch = tuple(t.to(device) for t in batch_data)
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
        }
        example_indices = batch[3].cpu().detach().cpu().tolist()

        outputs = self(**inputs)
        start_logits, end_logits = outputs

        return torch.stack([start_logits, end_logits], dim=2).detach().cpu(), torch.tensor([0]), \
               {'example_indices': example_indices}

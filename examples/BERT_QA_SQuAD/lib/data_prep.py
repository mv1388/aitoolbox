import os
import torch
from transformers import squad_convert_examples_to_features
from transformers.data.processors.squad import SquadV1Processor, SquadV2Processor


def load_and_cache_examples(tokenizer,
                            data_dir, predict_file, train_file,
                            model_name_or_path,
                            max_seq_length, max_query_length, doc_stride,
                            overwrite_cache=True, version_2_with_negative=True,
                            evaluate=False, output_examples=False,
                            num_examples=0):
    # Load data features from cache or dataset file
    input_dir = data_dir
    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, model_name_or_path.split("/"))).pop(),
            str(max_seq_length),
        ),
    )

    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file) and not overwrite_cache and not output_examples:
        print("Loading features from cached file %s", cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        features, dataset = features_and_dataset["features"], features_and_dataset["dataset"]
    else:
        print("Creating features from dataset file at %s", input_dir)

        processor = SquadV2Processor() if version_2_with_negative else SquadV1Processor()
        if evaluate:
            examples = processor.get_dev_examples(data_dir, filename=predict_file)
        else:
            examples = processor.get_train_examples(data_dir, filename=train_file)

        if num_examples > 0:
            examples = examples[:num_examples]

        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_training=not evaluate,
            return_dataset="pt",
        )

    if output_examples:
        return dataset, examples, features
    return dataset

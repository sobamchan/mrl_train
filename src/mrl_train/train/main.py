"""
The system trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) on the SNLI + MultiNLI (AllNLI) dataset
with softmax loss function. At every 1000 training steps, the model is evaluated on the
STS benchmark dataset

Usage:
python training_nli.py

OR
python training_nli.py pretrained_transformer_model_name
"""

import logging
from pathlib import Path
import traceback
from dataclasses import dataclass
from argparse import ArgumentParser

from datasets import load_dataset

from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.similarity_functions import SimilarityFunction
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

from mrl_train.utils import read_toml


@dataclass(frozen=True)
class Config:
    model_name: str
    matryoshka_dims: list[int]
    output_dir: Path

    @classmethod
    def from_config(cls) -> "Config":
        parser = ArgumentParser()
        parser.add_argument("--config", type=str)
        args = parser.parse_args()
        config = read_toml(args.config)

        return cls(
            model_name=config["model_name"],
            matryoshka_dims=config["matryoshka_dims"],
            output_dir=config["output_dir"],
        )


def main():
    # Set the log level to INFO to get more information
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    config = Config.from_config()

    # You can specify any Hugging Face pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
    model_name = config.model_name
    train_batch_size = 16

    output_dir = config.output_dir

    # 1. Here we define our SentenceTransformer model. If not already a Sentence Transformer model, it will automatically
    # create one with "mean" pooling.
    model = SentenceTransformer(model_name)

    # 2. Load the AllNLI dataset: https://huggingface.co/datasets/sentence-transformers/all-nli
    # We'll start with 10k training samples, but you can increase this to get a stronger model
    logging.info("Read AllNLI train dataset")
    train_dataset = load_dataset(
        "sentence-transformers/all-nli", "triplet", split="train"
    )
    eval_dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="dev")
    logging.info(train_dataset)

    # 3. Define our training loss: https://sbert.net/docs/package_reference/sentence_transformer/losses.html#softmaxloss
    base_train_loss = losses.MultipleNegativesRankingLoss(model=model)
    train_loss = losses.MatryoshkaLoss(
        model=model, loss=base_train_loss, matryoshka_dims=config.matryoshka_dims
    )

    # 4. Define an evaluator for use during training. This is useful to keep track of alongside the evaluation loss.
    stsb_eval_dataset = load_dataset("sentence-transformers/stsb", split="validation")
    dev_evaluator = EmbeddingSimilarityEvaluator(
        sentences1=stsb_eval_dataset["sentence1"],
        sentences2=stsb_eval_dataset["sentence2"],
        scores=stsb_eval_dataset["score"],
        main_similarity=SimilarityFunction.COSINE,
        name="sts-dev",
    )
    logging.info("Evaluation before training:")
    dev_evaluator(model)

    # 5. Define the training arguments
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=output_dir,
        # Optional training parameters:
        num_train_epochs=5,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=train_batch_size,
        warmup_ratio=0.1,
        fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=False,  # Set to True if you have a GPU that supports BF16
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        logging_steps=100,
        run_name="nli-v1",  # Will be used in W&B if `wandb` is installed
    )

    # 6. Create the trainer & start training
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=train_loss,
        evaluator=dev_evaluator,
    )
    trainer.train()

    # 7. Evaluate the model performance on the STS Benchmark test dataset
    test_dataset = load_dataset("sentence-transformers/stsb", split="test")
    test_evaluator = EmbeddingSimilarityEvaluator(
        sentences1=test_dataset["sentence1"],
        sentences2=test_dataset["sentence2"],
        scores=test_dataset["score"],
        main_similarity=SimilarityFunction.COSINE,
        name="sts-test",
    )
    test_evaluator(model)

    # 8. Save the trained & evaluated model locally
    final_output_dir = f"{output_dir}/final"
    model.save(final_output_dir)

    # 9. (Optional) save the model to the Hugging Face Hub!
    # It is recommended to run `huggingface-cli login` to log into your Hugging Face account first
    # model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
    # try:
    #     model.push_to_hub(f"{model_name}-nli-v1")
    # except Exception:
    #     logging.error(
    #         f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, you can run "
    #         f"`huggingface-cli login`, followed by loading the model using `model = SentenceTransformer({final_output_dir!r})` "
    #         f"and saving it using `model.push_to_hub('{model_name}-nli-v1')`."
    #     )


if __name__ == "__main__":
    main()

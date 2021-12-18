import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    AutoModelForQuestionAnswering, Trainer, TrainingArguments, HfArgumentParser
from helpers import prepare_dataset_nli, prepare_train_dataset_qa, \
    prepare_validation_dataset_qa, QuestionAnsweringTrainer, compute_accuracy
import os
import json

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import SequentialSampler
# import textattack

# import checklist
# from checklist.editor import Editor
# from checklist.perturb import Perturb
import random

import nltk

nltk.download('wordnet')
from nltk.corpus import wordnet

NUM_PREPROCESSING_WORKERS = 2


def query_wordnet(word):
    synonyms, antonyms, hypernyms, hyponyms = [], [], [], []

    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            synonyms.append(lemma.name())
            if lemma.antonyms():
                antonyms.append(lemma.antonyms()[0].name())

        hypernyms.extend([x.name().partition(".")[0] for x in synset.hypernyms()])
        hyponyms.extend([x.name().partition(".")[0] for x in synset.hyponyms()])

    return synonyms, antonyms, hypernyms, hyponyms


def augmenting_swap(example, max_returned=10):
    """
    A function to augment a hypothesis based on word replacement, using the above rules.
    """

    # import random
    # import time # To time augmentation of examples

    # start = time.time()

    hypo_sentence = example["hypothesis"][0]
    hypo = hypo_sentence.split(" ")
    label = example["label"][0]

    new_hypos = []
    new_labels = []
    num_aug_exs = 0

    if label == 0:
        # Entailment
        for idx, word in enumerate(hypo):

            synonyms, antonyms, hypernyms, hyponyms = query_wordnet(word)

            if synonyms:
                synonym = random.choice(synonyms)
                new_hypo = hypo[:idx] + [synonym]
                if idx != len(hypo) - 1:
                    new_hypo += hypo[(idx + 1):]
                new_hypos += [' '.join(new_hypo)]
                new_labels.append(0)
                num_aug_exs += 1
                if num_aug_exs == max_returned:
                    break

            if hypernyms:
                hypernym = random.choice(hypernyms)
                new_hypo = hypo[:idx] + [hypernym]
                if idx != len(hypo) - 1:
                    new_hypo += hypo[(idx + 1):]
                new_hypos += [' '.join(new_hypo)]
                new_labels.append(0)
                num_aug_exs += 1
                if num_aug_exs == max_returned:
                    break

            if antonyms:
                antonym = random.choice(antonyms)
                new_hypo = hypo[:idx] + [antonym]
                if idx != len(hypo) - 1:
                    new_hypo += hypo[(idx + 1):]
                new_hypos += [' '.join(new_hypo)]
                new_labels.append(2)
                num_aug_exs += 1
                if num_aug_exs == max_returned:
                    break

            if hyponyms:
                hyponym = random.choice(hyponyms)
                new_hypo = hypo[:idx] + [hyponym]
                if idx != len(hypo) - 1:
                    new_hypo += hypo[(idx + 1):]
                new_hypos += [' '.join(new_hypo)]
                new_labels.append(1)
                num_aug_exs += 1
                if num_aug_exs == max_returned:
                    break

    elif label == 1:
        # Neutral
        for idx, word in enumerate(hypo):

            synonyms, antonyms, hypernyms, hyponyms = query_wordnet(word)

            if synonyms:
                synonym = random.choice(synonyms)
                new_hypo = hypo[:idx] + [synonym]
                if idx != len(hypo) - 1:
                    new_hypo += hypo[(idx + 1):]
                new_hypos += [' '.join(new_hypo)]
                new_labels.append(1)
                num_aug_exs += 1
                if num_aug_exs == max_returned:
                    break

            if hypernyms:
                hypernym = random.choice(hypernyms)
                new_hypo = hypo[:idx] + [hypernym]
                if idx != len(hypo) - 1:
                    new_hypo += hypo[(idx + 1):]
                new_hypos += [' '.join(new_hypo)]
                new_labels.append(1)
                num_aug_exs += 1
                if num_aug_exs == max_returned:
                    break

    else:
        # Negation
        for idx, word in enumerate(hypo):

            synonyms, antonyms, hypernyms, hyponyms = query_wordnet(word)

            if synonyms:
                synonym = random.choice(synonyms)
                new_hypo = hypo[:idx] + [synonym]
                if idx != len(hypo) - 1:
                    new_hypo += hypo[(idx + 1):]
                new_hypos += [' '.join(new_hypo)]
                new_labels.append(2)
                num_aug_exs += 1
                if num_aug_exs == max_returned:
                    break

            if hypernyms:
                hypernym = random.choice(hypernyms)
                new_hypo = hypo[:idx] + [hypernym]
                if idx != len(hypo) - 1:
                    new_hypo += hypo[(idx + 1):]
                new_hypos += [' '.join(new_hypo)]
                new_labels.append(2)
                num_aug_exs += 1
                if num_aug_exs == max_returned:
                    break

            if hyponyms:
                hyponym = random.choice(hyponyms)
                new_hypo = hypo[:idx] + [hyponym]
                if idx != len(hypo) - 1:
                    new_hypo += hypo[(idx + 1):]
                new_hypos += [' '.join(new_hypo)]
                new_labels.append(2)
                num_aug_exs += 1
                if num_aug_exs == max_returned:
                    break

    # end = time.time()

    # print("Augmentation finished in {:.3f} sec".format(end - start))

    return new_hypos, new_labels


# def augment_data(example):

#     augmenter = textattack.augmentation.CheckListAugmenter(high_yield=True, fast_augment=True, transformations_per_example=2)

#     premise_outputs = []
#     hypothesis_outputs = []
#     label_outputs = []
#     for i in range(len(example['label'])):

#         augmented_premise = augmenter.augment(example["premise"][i])
#         premise_outputs += [example["premise"][i]] + augmented_premise

#         augmented_hypothesis = augmenter.augment(example["hypothesis"][i])
#         ypothesis_outputs += [example["hypothesis"]][i] + augmented_hypothesis

#         if len(ise_outputs = e) < len(hypothesis_outputs):
#             hypothesis_outputs = hypothesis_outputs[0:len(premise_outputs)]
#             label_outputs = [example["label"][i]] * len(premise_outputs)
#         elif len(premise_outputs) > len(hypothesis_outputs):
#             premise_outputs = premise_outputs[0:len(hypothesis_outputs)]
#             label_outputs = [example["label"][i]] * len(hypothesis_outputs)
#         else:
#             label_outputs = [example["label"][i]] * len(hypothesis_outputs)

#     return {"premise": premise_outputs, "hypothesis": hypothesis_outputs, "label": label_outputs}


def augment_data_word_swap(example):
    new_hypos, new_labels = augmenting_swap(example)
    cat_premises = example["premise"] * len(new_labels)
    return {"premise": cat_premises, "hypothesis": new_hypos, "label": new_labels}


class CustomTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super(CustomTrainer, self).__init__(*args, **kwargs)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.loss_func = nn.CrossEntropyLoss().to(device)  # weight=torch.Tensor([0.25, 0.5, 0.25]).to(device), label_smoothing=0.1
        self.epsilon: float = 0.1
        self.ignore_index: int = -100

    def compute_loss(self, model, inputs, return_outputs=False):

        labels = inputs.pop("labels")
        outputs = model(**inputs)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # Cross Entropy Loss

        logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
        loss_ce = self.loss_func(logits, labels)

        # Default Trainer NLLoss

        log_probs = -F.log_softmax(logits, dim=-1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)
        padding_mask = labels.eq(self.ignore_index)
        labels = torch.clamp(labels, min=0)
        nll_loss = log_probs.gather(dim=-1, index=labels)
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)
        nll_loss.masked_fill_(padding_mask, 0.0)
        smoothed_loss.masked_fill_(padding_mask, 0.0)
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        nll_loss = nll_loss.sum() / num_active_elements
        smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
        loss_nll = (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss

        # Weighted loss

        loss = 0.5 * loss_nll + 0.5 * loss_ce

        return (loss, outputs) if return_outputs else loss

    def _get_train_sampler(self):
        # By overriding this we create a sequential batch sampling method, disabling shuffling in training.
        # This will ensure that perturbed examples are seen together during training because they have
        # adjacent indices in the Dataset, enabling Contrastive Learning (ref. paper by Dua et al)
        return SequentialSampler(data_source=self.train_dataset)


def main():
    argp = HfArgumentParser(TrainingArguments)
    # The HfArgumentParser object collects command-line arguments into an object (and provides default values for unspecified arguments).
    # In particular, TrainingArguments has several keys that you'll need/want to specify (when you call run.py from the command line):
    # --do_train
    #     When included, this argument tells the script to train a model.
    #     See docstrings for "--task" and "--dataset" for how the training dataset is selected.
    # --do_eval
    #     When included, this argument tells the script to evaluate the trained/loaded model on the validation split of the selected dataset.
    # --per_device_train_batch_size <int, default=8>
    #     This is the training batch size.
    #     If you're running on GPU, you should try to make this as large as you can without getting CUDA out-of-memory errors.
    #     For reference, with --max_length=128 and the default ELECTRA-small model, a batch size of 32 should fit in 4gb of GPU memory.
    # --num_train_epochs <float, default=3.0>
    #     How many passes to do through the training data.
    # --output_dir <path>
    #     Where to put the trained model checkpoint(s) and any eval predictions.
    #     *This argument is required*.

    argp.add_argument('--model', type=str,
                      default='google/electra-small-discriminator',
                      help="""This argument specifies the base model to fine-tune.
        This should either be a HuggingFace model ID (see https://huggingface.co/models)
        or a path to a saved model checkpoint (a folder containing config.json and pytorch_model.bin).""")
    argp.add_argument('--task', type=str, choices=['nli', 'qa'], required=True,
                      help="""This argument specifies which task to train/evaluate on.
        Pass "nli" for natural language inference or "qa" for question answering.
        By default, "nli" will use the SNLI dataset, and "qa" will use the SQuAD dataset.""")
    argp.add_argument('--dataset', type=str, default=None,
                      help="""This argument overrides the default dataset used for the specified task.""")
    argp.add_argument('--max_length', type=int, default=128,
                      help="""This argument limits the maximum sequence length used during training/evaluation.
        Shorter sequence lengths need less memory and computation time, but some examples may end up getting truncated.""")
    argp.add_argument('--max_train_samples', type=int, default=None,
                      help='Limit the number of examples to train on.')
    argp.add_argument('--max_eval_samples', type=int, default=None,
                      help='Limit the number of examples to evaluate on.')

    training_args, args = argp.parse_args_into_dataclasses()

    # Dataset selection
    default_datasets = {'qa': ('squad',), 'nli': ('snli',)}
    dataset_id = tuple(args.dataset.split(':')) if args.dataset is not None else \
        default_datasets[args.task]
    # NLI models need to have the output label count specified (label 0 is "entailed", 1 is "neutral", and 2 is "contradiction")
    task_kwargs = {'num_labels': 3} if args.task == 'nli' else {}
    # MNLI has two validation splits (one with matched domains and one with mismatched domains). Most datasets just have one "validation" split
    eval_split = 'validation_matched' if dataset_id == ('glue', 'mnli') else 'validation'
    # Load the raw data
    dataset = datasets.load_dataset(*dataset_id)

    # Here we select the right model fine-tuning head
    model_classes = {'qa': AutoModelForQuestionAnswering,
                     'nli': AutoModelForSequenceClassification}
    model_class = model_classes[args.task]
    # Initialize the model and tokenizer from the specified pretrained model/checkpoint
    model = model_class.from_pretrained(args.model, **task_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    # Select the dataset preprocessing function (these functions are defined in helpers.py)
    if args.task == 'qa':
        prepare_train_dataset = lambda exs: prepare_train_dataset_qa(exs, tokenizer)
        prepare_eval_dataset = lambda exs: prepare_validation_dataset_qa(exs, tokenizer)
    elif args.task == 'nli':
        prepare_train_dataset = prepare_eval_dataset = \
            lambda exs: prepare_dataset_nli(exs, tokenizer, args.max_length)
        # prepare_eval_dataset = prepare_dataset_nli
    else:
        raise ValueError('Unrecognized task name: {}'.format(args.task))

    print("Preprocessing data... (this takes a little bit, should only happen once per dataset)")
    if dataset_id == ('snli',):
        # remove SNLI examples with no label
        dataset = dataset.filter(lambda ex: ex['label'] != -1)

    train_dataset = None
    eval_dataset = None
    train_dataset_featurized = None
    eval_dataset_featurized = None
    if training_args.do_train:
        train_dataset = dataset['train']
        if args.max_train_samples:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        # print("Training Set Length Before Augmentation: ", len(train_dataset))
        # editor = Editor()
        augmented_train_dataset = train_dataset.map(augment_data_word_swap, batched=True, batch_size=1,
                                                    remove_columns=train_dataset.column_names)
        # print("Training Set Length After Augmentation: ", len(augmented_train_dataset)) # this forces it to do augmentation before, I think it does it lazily when not forced
        train_dataset_featurized = augmented_train_dataset.map(
            prepare_train_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=augmented_train_dataset.column_names
        )
    if training_args.do_eval:
        eval_dataset = dataset[eval_split]
        if args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        eval_dataset_featurized = eval_dataset.map(
            prepare_eval_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=eval_dataset.column_names
        )

    # Select the training configuration
    trainer_class = CustomTrainer
    eval_kwargs = {}
    # If you want to use custom metrics, you should define your own "compute_metrics" function.
    # For an example of a valid compute_metrics function, see compute_accuracy in helpers.py.
    compute_metrics = None
    if args.task == 'qa':
        # For QA, we need to use a tweaked version of the Trainer (defined in helpers.py)
        # to enable the question-answering specific evaluation metrics
        trainer_class = QuestionAnsweringTrainer
        eval_kwargs['eval_examples'] = eval_dataset
        metric = datasets.load_metric('squad')
        compute_metrics = lambda eval_preds: metric.compute(
            predictions=eval_preds.predictions, references=eval_preds.label_ids)
    elif args.task == 'nli':
        compute_metrics = compute_accuracy

    # This function wraps the compute_metrics function, storing the model's predictions
    # so that they can be dumped along with the computed metrics
    eval_predictions = None

    def compute_metrics_and_store_predictions(eval_preds):
        nonlocal eval_predictions
        eval_predictions = eval_preds
        return compute_metrics(eval_preds)

    # Initialize the Trainer object with the specified arguments and the model and dataset we loaded above
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset_featurized,
        eval_dataset=eval_dataset_featurized,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_and_store_predictions
    )
    # This is to confirm that the batch-sequential sampler is being used, see Trainer method overoads above
    # print(trainer.get_train_dataloader().sampler)

    # Train and/or evaluate
    if training_args.do_train:
        trainer.train()
        trainer.save_model()
        # If you want to customize the way the loss is computed, you should subclass Trainer and override the "compute_loss"
        # method (see https://huggingface.co/transformers/_modules/transformers/trainer.html#Trainer.compute_loss).
        #
        # You can also add training hooks using Trainer.add_callback:
        #   See https://huggingface.co/transformers/main_classes/trainer.html#transformers.Trainer.add_callback
        #   and https://huggingface.co/transformers/main_classes/callback.html#transformers.TrainerCallback

    if training_args.do_eval:
        results = trainer.evaluate(**eval_kwargs)

        # To add custom metrics, you should replace the "compute_metrics" function (see comments above).
        #
        # If you want to change how predictions are computed, you should subclass Trainer and override the "prediction_step"
        # method (see https://huggingface.co/transformers/_modules/transformers/trainer.html#Trainer.prediction_step).
        # If you do this your custom prediction_step should probably start by calling super().prediction_step and modifying the
        # values that it returns.

        print('Evaluation results:')
        print(results)

        os.makedirs(training_args.output_dir, exist_ok=True)

        with open(os.path.join(training_args.output_dir, 'eval_metrics.json'), encoding='utf-8', mode='w') as f:
            json.dump(results, f)

        with open(os.path.join(training_args.output_dir, 'eval_predictions.jsonl'), encoding='utf-8', mode='w') as f:
            if args.task == 'qa':
                predictions_by_id = {pred['id']: pred['prediction_text'] for pred in eval_predictions.predictions}
                for example in eval_dataset:
                    example_with_prediction = dict(example)
                    example_with_prediction['predicted_answer'] = predictions_by_id[example['id']]
                    f.write(json.dumps(example_with_prediction))
                    f.write('\n')
            else:
                for i, example in enumerate(eval_dataset):
                    example_with_prediction = dict(example)
                    example_with_prediction['predicted_scores'] = eval_predictions.predictions[i].tolist()
                    example_with_prediction['predicted_label'] = int(eval_predictions.predictions[i].argmax())
                    f.write(json.dumps(example_with_prediction))
                    f.write('\n')


if __name__ == "__main__":
    main()

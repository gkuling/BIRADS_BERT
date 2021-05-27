import argparse
import os
import torch
import logging
import random
import numpy as np
from transformers import BertConfig, BertForMaskedLM, AdamW, \
    get_linear_schedule_with_warmup, BertTokenizer
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from tqdm import tqdm, trange
from tokenizers.implementations import BertWordPieceTokenizer
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.data.datasets import TextDataset

from datetime import datetime as dt

tic = dt.now()

parser = argparse.ArgumentParser()
logger = logging.getLogger(__name__)
# Required parameters
parser.add_argument("--train_data_file", default=None, type=str,
                    required=True,
                    help="The input training data in a .txt file"
                         "files.")
parser.add_argument("--output_dir", default=None, type=str, required=True,
                    help="The output directory where the model predictions "
                         "and checkpoints will be written.")
parser.add_argument('--overwrite_output_dir', action='store_true',
                    help="Overwrite the content of the output directory")
parser.add_argument("--per_gpu_train_batch_size", default=16, type=int,
                    help="Batch size per GPU/CPU for training.")
parser.add_argument("--do_eval", action='store_true',
                    help="Whether to run eval on the dev set.")
parser.add_argument("--eval_data_file", default=None, type=str,
                    required=False,
                    help="The input training data in a .txt file"
                         "files.")
parser.add_argument("--num_train_epochs", default=1.0, type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--warmup_steps", default=2000, type=int,
                    help="Linear warmup over warmup_steps.")
parser.add_argument('--save_steps', type=int, default=10000,
                    help="Save checkpoint every X updates steps.")
parser.add_argument('--data_portion', type=float, default=1.0,
                    help="The portion of the training data you wish to load. "
                         "(1.0 for all data, >1.0 for a portion")
parser.add_argument('--logging_steps', type=int, default=10000,
                    help="Log every X updates steps.")
parser.add_argument('--block_size', type=int, default=32,
                    help="Max sequence length used in tokenizer and dataset.")
parser.add_argument("--start_from_checkpoint", action='store_true',
                    help="Start training from latest checkpoint.")
parser.add_argument("--preliminary_model", type=str, default='fromScratch',
                    help='Choice to start the model from a previously trained '
                         'model or start from scratch. Used with '
                         'model.from_pretrained(preliminary_model. ')
args = parser.parse_args()

def set_seed(sd):
    random.seed(sd)
    np.random.seed(sd)
    torch.manual_seed(sd)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(sd)

def evaluate(args, model, eval_dataset, tokenizer, step, prefix=""):
    """
    Evaluation of model
    :param args: input arguments from parser
    :param model: pytorch model to be evaluated
    :param eval_dataset: dataset used for evaluation
    :param tokenizer: tokenizer used by the model
    :param step: the current step in training
    :param prefix: prescript to be added to the beginning of save file
    :return: results of evaluation
    """
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    print('')
    eval_output_dir = args.output_dir

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    eval_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                 batch_size=eval_batch_size,
                                 collate_fn=data_collator
                                 )

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.train_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in tqdm(eval_dataloader,
                      desc="Evaluating",
                      position=0,
                      leave=True):

        with torch.no_grad():
            outputs = model(input_ids=batch['input_ids'].to(args.device),
                            labels=batch['labels'].to(args.device))
            loss = outputs['loss']
            eval_loss += loss.mean().item()

        nb_eval_steps += 1

    eval_loss /= nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {
        "perplexity": perplexity,
        'loss': eval_loss,
        "Iteration": str(step)
    }

    output_eval_file = os.path.join(eval_output_dir, prefix,
                                    "eval_results.txt")
    with open(output_eval_file, "a") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        writer.write('\n')
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s, " % (key, str(result[key])))

    writer.close()

    return result

def train(args, train_dataset, model, tokenizer, eval_dataset=None):
    """
     Train the model
    :param args: input arguments from parser
    :param train_dataset: dataset used for training
    :param model: pytorch model to be evaluated
    :param tokenizer: tokenizer used by the model
    :param eval_dataset: dataset used for evaluation
    :return:
    """

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size,
                                  collate_fn=data_collator
                                  )

    init_total = len(
        train_dataloader) * args.num_train_epochs

    # loading a modle from a checkpoint if neccesary
    if args.start_from_checkpoint:
        chk_pt_fdlr = [fldr for fldr in os.listdir(args.output_dir) if
                       fldr.startswith('checkpoint')]
        chk_pt_fdlr.sort()
        logger.info("***** Running training from checkpoint: " + str(
            chk_pt_fdlr[-1]) + "*****")
        global_step = int(''.join([chr for chr in chk_pt_fdlr[-1]
                                   if chr.isdigit()]))
        it_total = init_total - global_step
        args.num_train_epochs = np.round(it_total / len(train_dataloader))
        # model = BertForMaskedLM(config=config)
        model = BertForMaskedLM.from_pretrained(args.output_dir + '/' +
                                                chk_pt_fdlr[-1])
        model.to(args.device)

        logger.info('Loaded checkpoint model. Beginning training.')
    else:
        global_step = 0
        it_total = init_total

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if
                    not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if
                    any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5,
                      eps=1e-8)
    if global_step > args.warmup_steps:
        scheduler = \
            get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=args.warmup_steps,
                                            num_training_steps=init_total)
        for _ in range(global_step):
            scheduler.step()
        logger.info('Initialized LR Scheduler and brought it to current step.')
    else:
        scheduler = \
            get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=args.warmup_steps,
                                            num_training_steps=it_total)
    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info("  Total optimization steps = %d", it_total)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(seed)  # Added here for reproducibility (even between python 2
    # and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader,
                              desc="Iteration",
                              position=0,
                              leave=True)
        epoch_iterator.set_postfix({'loss': 'Initialized'})
        for step, batch in enumerate(epoch_iterator):
            model.train()
            outputs = model(input_ids=batch['input_ids'].to(args.device),
                            labels=batch['labels'].to(args.device))
            # model outputs are always tuple in transformers (see doc)
            loss = outputs['loss']

            if args.n_gpu > 1:
                # mean() to average on multi-gpu parallel training
                loss = loss.mean()

            loss.backward()

            tr_loss += loss.item()
            epoch_iterator.set_postfix({'loss': loss.item()})

            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           1.0)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1

            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                # Log metrics
                results = evaluate(args, model, eval_dataset, tokenizer,
                                   step=global_step)

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                checkpoint_prefix = 'checkpoint'
                # Save model checkpoint
                output_dir = os.path.join(args.output_dir,
                                          '{}-{}'.format(checkpoint_prefix,
                                                         global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = model.module \
                    if hasattr(model, 'module') \
                    else model  # Take care of distributed/parallel training
                model_to_save.save_pretrained(output_dir)
                torch.save(args,
                           os.path.join(output_dir, 'training_args.bin'))
                logger.info("Saving model checkpoint to %s", output_dir)

    return global_step, tr_loss / global_step, model


args.mlm = True

if os.path.exists(args.output_dir) and os.listdir(
        args.output_dir) and not args.overwrite_output_dir:
    raise ValueError(
        "Output directory ({}) already exists and is not empty. Use "
        "--overwrite_output_dir to overcome.".format(
            args.output_dir))

# Setup CUDA, GPU & distributed training
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu")
args.n_gpu = torch.cuda.device_count()

args.device = device

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)
logger.info(
    "Device: %s, n_gpu: %s", device, args.n_gpu)

# Set seed
seed = 20210325
set_seed(seed)

logger.info("Beginning Tokenizer Training on data in " + args.train_data_file)
paths = args.train_data_file
args.vocab_size = int(''.join([char for char in args.train_data_file.split(
    '/')[-1] if char.isnumeric()]))
if not args.preliminary_model != 'fromScratch' and \
        not args.start_from_checkpoint:
    # Building custom Tokenizer
    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        strip_accents=True,
        lowercase=True,
    )
    tokenizer.train(
        paths,
        vocab_size=args.vocab_size + 5,
        min_frequency=2,
        show_progress=True,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        limit_alphabet=1000,
        wordpieces_prefix="##",
    )
    tokenizer.save_model(args.output_dir)

if args.preliminary_model != 'fromScratch':
    tokenizer = BertTokenizer.from_pretrained(args.preliminary_model)
else:
    tokenizer = BertTokenizer.from_pretrained(args.output_dir)

config = BertConfig.from_pretrained('bert-base-cased')
config.vocab_size = tokenizer.vocab_size
if args.preliminary_model != 'fromScratch':
    model = BertForMaskedLM.from_pretrained(args.preliminary_model)
else:
    model = BertForMaskedLM(config=config)
model.to(args.device)

train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=args.train_data_file,
    block_size=32,
    overwrite_cache=args.overwrite_output_dir
)

eval_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=args.eval_data_file,
    block_size=32,
    overwrite_cache=args.overwrite_output_dir
)
if args.data_portion < 1.0:
    train_dataset.examples = train_dataset.examples[:int(len(
        train_dataset.examples)*args.data_portion)]
    eval_dataset.examples = eval_dataset.examples[:int(len(
        eval_dataset.examples)*args.data_portion)]
    logger.info("Training and validation set limited to " + str(
        args.data_portion) + " portion of original data.")

logger.info("Training/evaluation parameters %s", args)

global_step, tr_loss, model = train(args,
                                    train_dataset,
                                    model,
                                    tokenizer,
                                    eval_dataset=eval_dataset)
logger.info(" global_step = %s, average loss = %s", global_step,
            tr_loss)

# Do the saving
# Create output directory if needed
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

logger.info("Saving model checkpoint to %s", args.output_dir)
# Save a trained model, configuration and tokenizer using `save_pretrained()`.
# They can then be reloaded using `from_pretrained()`
# Take care of parallel training
model_to_save = model.module if hasattr(model,
                                        'module') else model
model_to_save.save_pretrained(args.output_dir)

# Good practice: save your training arguments together with the trained model
torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

# Load a trained model and vocabulary that you have fine-tuned
model = BertForMaskedLM.from_pretrained(args.output_dir)
if args.preliminary_model != 'fromScratch':
    tokenizer = BertTokenizer.from_pretrained(args.preliminary_model)
else:
    tokenizer = BertTokenizer.from_pretrained(args.output_dir)
model.to(args.device)

# Evaluation
results = {}
if args.do_eval:
    checkpoints = [args.output_dir]

    logger.info("Evaluate the following checkpoints: %s", checkpoints)
    for checkpoint in checkpoints:
        global_step = checkpoint.split('-')[-1] if len(
            checkpoints) > 1 else ""
        prefix = checkpoint.split('/')[-1] if checkpoint.find(
            'checkpoint') != -1 else ""

        model = BertForMaskedLM.from_pretrained(checkpoint)
        model.to(args.device)
        result = evaluate(args, model, eval_dataset, tokenizer, step='TestSet')
        result = dict(
            (k + '_{}'.format(global_step), v) for k, v in result.items())
        results.update(result)
toc = dt.now()
print("End of MLM_Training_transformers.py Script.")
print('Total Script Runtime: ' + str(toc-tic))

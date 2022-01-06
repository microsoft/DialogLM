# DialogLM (without sparse attention)
## Environment

The environment for UniLM refers to [this link](https://github.com/microsoft/unilm/tree/master/s2s-ft).

We recommend to use the transformers library of version 2.7.0 (`pip install --user transformers==2.7.0`).

## Pre-trained Models

Pre-trained Model (**DialogLM**) can be downloaded from [here](https://drive.google.com/file/d/18_RhUhGh_pJuQXGpXretbqEWrHBNNuhb/view?usp=sharing).

## Input File Format

We support two dataset formats:

1. Text format: each line contains a json string of an example. `"src"` contains source sequence text, `"tgt"` contains target sequence text (`"tgt"` can be ignored for decoding). The data should be pre-processed as follows:

```bash
{"src": "Messages posted on social media claimed the user planned to `` kill as many people as possible ''", "tgt": "Threats to kill pupils in a shooting at a Blackpool school are being investigated by Lancashire police ."}
{"src": "Media playback is unsupported on your device", "tgt": "A slide running the entire length of one of the steepest city centre streets in Europe has been turned into a massive three-lane water adventure ."}
{"src": "Chris Erskine crossed low for Kris Doolan to tap home and give the Jags an early lead .", "tgt": "Partick Thistle will finish in the Scottish Premiership 's top six for the first time after beating Motherwell"}
```

2. Tokenized format: if you use tokenized data (with the same WordPiece tokenizers as BERT), `"src"` is a list of source sequence tokens, and `"tgt"` is a list of target sequence tokens (`"tgt"` can be ignored for decoding):

```bash
{"src": ["messages", "posted", "on", "social", "media", "claimed", "the", "user", "planned", "to", "\"", "kill", "as", "many", "people", "as", "possible", "\""], "tgt": ["threats", "to", "kill", "pupils", "in", "a", "shooting", "at", "a", "blackpool", "school", "are", "being", "investigated", "by", "lancashire", "police", "."]}
{"src": ["media", "playback", "is", "un", "##su", "##pp", "##orted", "on", "your", "device"], "tgt": ["a", "slide", "running", "the", "entire", "length", "of", "one", "of", "the", "steep", "##est", "city", "centre", "streets", "in", "europe", "has", "been", "turned", "into", "a", "massive", "three", "-", "lane", "water", "adventure", "."]}
{"src": ["chris", "erskine", "crossed", "low", "for", "kris", "doo", "##lan", "to", "tap", "home", "and", "give", "the", "ja", "##gs", "an", "early", "lead", "."], "tgt": ["part", "##ick", "thistle", "will", "finish", "in", "the", "scottish", "premiership", "'", "s", "top", "six", "for", "the", "first", "time", "after", "beating", "mother", "##well"]}
```

The code automatically detects the input format. If the json line contains `list`, we process the input as the tokenized format; if the json line contains `string`, the code will tokenize them.


## Example: AMI with DialogLM

### Fine-tuning

Pre-processed json dataset path: user/mzhong/data/AMI

```bash
# path of training data
TRAIN_FILE=/your/path/to/train.json
# folder used to save fine-tuned checkpoints
OUTPUT_DIR=/your/path/to/save_checkpoints
# folder used to cache package dependencies
CACHE_DIR=/your/path/to/transformer_package_cache

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m torch.distributed.launch --nproc_per_node=8 run_seq2seq.py \
  --train_file ${TRAIN_FILE} --output_dir ${OUTPUT_DIR} \
  --model_type unilm --model_name_or_path ${MODEL_PATH}/pytorch_model.bin \
  --config_name ${MODEL_PATH}/config.json --tokenizer_name unilm1.2-base-uncased \
  --do_lower_case --fp16 --fp16_opt_level O2 --max_source_seq_length 5120 --max_target_seq_length 512 \
  --per_gpu_train_batch_size 1 --gradient_accumulation_steps 8 \
  --learning_rate 7e-5 --num_warmup_steps 500 --num_training_steps 5000 --save_steps 500 --cache_dir ${CACHE_DIR}
```

- The fine-tuning batch size = `number of gpus` * `per_gpu_train_batch_size` * `gradient_accumulation_steps`. So in the above example, the batch size is `8*1*8 = 64`. The three arguments need to be adjusted together in order to remain the total batch size unchanged.
- `--do_lower_case`: for uncased models

### Decoding and Evaluation

Please put the golden answer text files (e.g., test.target, one reference summary per line) in the data folder.

```bash
# path of the fine-tuned checkpoint
MODEL_PATH=/your/path/to/model_checkpoint
SPLIT=test
# input file that you would like to decode
INPUT_JSON=/your/path/to/${SPLIT}.json

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

python decode_seq2seq.py \
  --fp16 --model_type unilm --tokenizer_name unilm1.2-base-uncased --input_file ${INPUT_JSON} --split $SPLIT --do_lower_case \
  --model_path ${MODEL_PATH} --max_seq_length 5632 --max_tgt_length 512 --batch_size 8 --beam_size 6 \
  --mode s2s --forbid_ignore_word "." --forbid_duplicate_ngrams --min_len 256 --length_penalty 1.0 \

GOLD_PATH=/your/path/to/${SPLIT}.target
# ${MODEL_PATH}.${SPLIT} is the predicted target file
python evaluations/eval_for_xsum.py --pred ${MODEL_PATH}.${SPLIT} --gold ${GOLD_PATH} --split ${SPLIT}
```

- The decoding results are saved at `${MODEL_PATH}.${SPLIT}`.
- `--do_lower_case`: for uncased models
- `max_seq_length` = `max_source_seq_length` + `max_tgt_length`

## Example: TVMegaSite with DialogLM

### Fine-tuning

Pre-processed json dataset path: user/mzhong/data/TVMegaSite

```bash
# path of training data
TRAIN_FILE=/your/path/to/train.json
# folder used to save fine-tuned checkpoints
OUTPUT_DIR=/your/path/to/save_checkpoints
# folder used to cache package dependencies
CACHE_DIR=/your/path/to/transformer_package_cache

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m torch.distributed.launch --nproc_per_node=8 run_seq2seq.py \
  --train_file ${TRAIN_FILE} --output_dir ${OUTPUT_DIR} \
  --model_type unilm --model_name_or_path ${MODEL_PATH}/pytorch_model.bin \
  --config_name ${MODEL_PATH}/config.json --tokenizer_name unilm1.2-base-uncased \
  --do_lower_case --fp16 --fp16_opt_level O2 --max_source_seq_length 5120 --max_target_seq_length 512 \
  --per_gpu_train_batch_size 1 --gradient_accumulation_steps 8 \
  --learning_rate 7e-5 --num_warmup_steps 2000 --num_training_steps 100000 --save_steps 5000 --cache_dir ${CACHE_DIR}
```

- The fine-tuning batch size = `number of gpus` * `per_gpu_train_batch_size` * `gradient_accumulation_steps`. So in the above example, the batch size is `8*1*8 = 64`. The three arguments need to be adjusted together in order to remain the total batch size unchanged.
- `--do_lower_case`: for uncased models

### Decoding and Evaluation

Please put the golden answer text files (e.g., test.target, one reference summary per line) in the data folder.

```bash
# path of the fine-tuned checkpoint
MODEL_PATH=/your/path/to/model_checkpoint
SPLIT=test
# input file that you would like to decode
INPUT_JSON=/your/path/to/${SPLIT}.json

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

python decode_seq2seq.py \
  --fp16 --model_type unilm --tokenizer_name unilm1.2-base-uncased --input_file ${INPUT_JSON} --split $SPLIT --do_lower_case \
  --model_path ${MODEL_PATH} --max_seq_length 5480 --max_tgt_length 360 --batch_size 20 --beam_size 6 \
  --length_penalty 1.0 --mode s2s --forbid_ignore_word "." --forbid_duplicate_ngrams --min_len 192

GOLD_PATH=/your/path/to/${SPLIT}.target
# ${MODEL_PATH}.${SPLIT} is the predicted target file
python evaluations/eval_for_xsum.py --pred ${MODEL_PATH}.${SPLIT} --gold ${GOLD_PATH} --split ${SPLIT}
```

- The decoding results are saved at `${MODEL_PATH}.${SPLIT}`.
- `--do_lower_case`: for uncased models
- `max_seq_length` = `max_source_seq_length` + `max_tgt_length`


# Language Modeling on One Billion Words

## Data
[One Billion Words](https://www.statmt.org/lm-benchmark/) with the [standard vocabulary](https://github.com/rafaljozefowicz/lm/blob/master/1b_word_vocab.txt). 

## Pre-Process
```base
python pre_word_ada/encode_data2folder.py --train_folder <train_folder> --test_folder <test_folder> --vocab <vocab_file> --output_folder <output_folder> 
```

## Training
For all experiments, the recommended random seeds are in {1, 101, 8191, 65537, 131071, 524287, 6700417}.
### Adam & RAdam
```base
python -u train_1bw.py --dataset_folder <data_folder> --epochs 20 \
    --opt [adam|radam] --lr 0.001 --milestone 12 18 --clip 1.0 \
    --model_path <model path> --run <run_id> --seed <random seed> 
```

### Apollo
```base
python -u train_1bw.py --dataset_folder <data_folder> --epochs 20 \
    --opt apollo --lr 10.0 --milestone 12 18 --clip 1.0 \
    --warmup_updates 400 --init_lr 0.01 \
    --model_path <model path> --run <run_id> --seed <random seed> 
```

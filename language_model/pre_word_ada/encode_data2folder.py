import pickle
import argparse
import os
import codecs
from tqdm import tqdm


def encode_dataset(input_folder, w_map):
    w_sos = w_map['<S>']
    w_eos = w_map['</S>']
    w_unk = w_map['<UNK>']

    list_dirs = os.walk(input_folder)

    lines = list()

    for root, dirs, files in list_dirs:
        for file in tqdm(files):
            if file.startswith('news.en.heldout'):
                with codecs.open(os.path.join(root, file), 'r', 'utf-8') as fin:
                    lines = lines + list(filter(lambda t: t and not t.isspace(), fin.readlines()))

    dataset = list()
    for line in lines:
        dataset += [w_sos] + list(map(lambda t: w_map.get(t, w_unk), line.strip().split())) + [w_eos]

    return dataset


def encode_dataset2file(input_folder, output_folder, w_map):
    w_sos = w_map['<S>']
    w_eos = w_map['</S>']
    w_unk = w_map['<UNK>']

    list_dirs = os.walk(input_folder)

    range_ind = 0

    for root, dirs, files in list_dirs:
        for file in tqdm(files):
            with codecs.open(os.path.join(root, file), 'r', 'utf-8') as fin:
                lines = list(filter(lambda t: t and not t.isspace(), fin.readlines()))

            dataset = []
            for line in lines:
                dataset.append([w_sos] + list(map(lambda t: w_map.get(t, w_unk), line.strip().split())) + [w_eos])

            with open(output_folder + 'train_' + str(range_ind) + '.pk', 'wb') as f:
                pickle.dump(dataset, f)

            range_ind += 1

    return range_ind


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_folder',
                        default="/data/billionwords/1-billion-word-language-modeling-benchmark/training-monolingual.tokenized.shuffled")
    parser.add_argument('--test_folder',
                        default="/data/billionwords/1-billion-word-language-modeling-benchmark/heldout-monolingual.tokenized.shuffled")
    parser.add_argument('--input_map', default="/data/billionwords/1b_map.pk")
    parser.add_argument('--output_folder', default="/data/billionwords/one_billion/")
    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--unk', default='<unk>')
    args = parser.parse_args()

    w_count = dict()
    with open(args.input_map, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            word = tokens[0]
            count = int(tokens[1])
            w_count[word] = count

    w_list = [(k, v) for k, v in w_count.items() if v > args.threshold]
    w_list.sort(key=lambda t: t[1], reverse=True)
    w_list.append(('</S>', w_count['<S>']))
    w_map = {kv[0]: v for v, kv in enumerate(w_list)}

    print('Vocab size: {}'.format(len(w_map)))

    range_ind = encode_dataset2file(args.train_folder, args.output_folder, w_map)

    test_dataset = encode_dataset(args.test_folder, w_map)

    with open(args.output_folder + 'test.pk', 'wb') as f:
        pickle.dump({'w_map': w_map, 'test_data': test_dataset, 'range': range_ind}, f)

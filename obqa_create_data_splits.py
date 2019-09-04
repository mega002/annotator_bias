
import argparse
import json
import pandas as pd


pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 1000)


def load_data():
    trn_file_path = "./OpenBookQA-V1-Sep2018/Data/Additional/train_complete.jsonl"
    dev_file_path = "./OpenBookQA-V1-Sep2018/Data/Additional/dev_complete.jsonl"

    examples = []
    with open(trn_file_path, "r") as fd:
        examples.extend(fd.readlines())
        num_trn_examples = len(examples)
        print("read {} training examples.".format(num_trn_examples))

    with open(dev_file_path, "r") as fd:
        examples.extend(fd.readlines())
        print("read {} development examples.".format(len(examples) - num_trn_examples))

    def parse_json_line(jline):
        line = json.loads(jline)
        parsed_line = {
            "id": line["id"],
            "turkIdAnonymized": line["turkIdAnonymized"],
            "answerKey": line["answerKey"],
            "clarity": line["clarity"],
            "fact1": line["fact1"],
            "humanScore": line["humanScore"],
            "question": line["question"]["stem"],
            "choice0": line["question"]["choices"][0]["text"],
            "choice1": line["question"]["choices"][1]["text"],
            "choice2": line["question"]["choices"][2]["text"],
            "choice3": line["question"]["choices"][3]["text"]
        }
        anskey_to_choice = {"A": "choice0", "B": "choice1", "C": "choice2", "D": "choice3"}
        parsed_line["correct_answer"] = parsed_line[anskey_to_choice[parsed_line["answerKey"]]]

        return parsed_line

    print("total number of examples: {}.".format(len(examples)))
    df = pd.DataFrame([parse_json_line(jline) for jline in examples])
    df = df.sample(frac=1.0)

    return df


def dump_to_json(example_set, split_type, split_partition, split_index, augment):
    example_set_filtered_cols = example_set.loc[:, ['id', 'turkIdAnonymized', 'question', 'correct_answer',
                                                    'answerKey', 'choice0', 'choice1', 'choice2', 'choice3']]

    data_dict = example_set_filtered_cols.reset_index().to_dict(orient='rows')
    file_name = split_partition + '_' + split_type + '_' + str(split_index) + augment + ".json"
    print('Saving %s with %d exmamples' % (file_name, len(data_dict)))
    with open(file_name, "w") as f:
        json.dump(data_dict, f, sort_keys=True, indent=4)


def print_annotator_distribution(data):
    dff = data.groupby(['turkIdAnonymized'])[
        [c for c in data.columns if c != 'turkIdAnonymized']].nunique().sort_values(
        'id', ascending=False).reset_index()
    total_number_of_examples = dff.id.sum()
    print("total number of unique examples: ", total_number_of_examples)
    dff["percentage"] = round(dff.id * 100.0 / total_number_of_examples, 1)
    print(dff)


def split_dev_sets(dev_sets, num_splits, augment_ratio):
    dev_sets_new = []
    dev_sets_move = []
    for i in range(num_splits):
        split_annotators = dev_sets[i]['turkIdAnonymized'].unique()
        split_annotator_splits = [
            dev_sets[i][dev_sets[i]['turkIdAnonymized'] == annotator]
            for annotator in split_annotators
        ]
        annotator_split_idx = [
            int(augment_ratio * len(split_annotator_splits[j]))
            for j in range(len(split_annotators))
        ]

        dev_set_new = []
        dev_set_move = []
        for j in range(len(split_annotators)):
            dev_set_new.append(split_annotator_splits[j][annotator_split_idx[j]:])
            dev_set_move.append(split_annotator_splits[j][0:annotator_split_idx[j]])

        dev_sets_new.append(pd.concat(dev_set_new))
        dev_sets_move.append(pd.concat(dev_set_move))

    dev_split_idx = [
        len(dev_sets_move[i])
        for i in range(num_splits)
    ]

    return dev_sets_new, dev_sets_move, dev_split_idx


def augment_data(train_sets, dev_sets, num_splits, augment_ratio,
                 by_annotator, dev_split_idx=None, normalize=True):
    if by_annotator:
        dev_sets_new, dev_sets_move, dev_split_idx = split_dev_sets(
            dev_sets, num_splits, augment_ratio)
    else:
        if not dev_split_idx:
            dev_split_idx = [
                int(augment_ratio * len(dev_sets[i]))
                for i in range(num_splits)
            ]
        dev_sets_new = [
            dev_sets[i][dev_split_idx[i]:]
            for i in range(num_splits)
        ]
        dev_sets_move = [
            dev_sets[i][0:dev_split_idx[i]]
            for i in range(num_splits)
        ]

    if normalize:
        augmented_train_sets = [
            train_sets[i].sample(n=len(train_sets[i]) - len(dev_sets_move[i]))
            for i in range(num_splits)
        ]
    else:
        augmented_train_sets = [
            train_sets[i].sample(frac=1.0)
            for i in range(num_splits)
        ]

    augmented_train_sets = [
        pd.concat([augmented_train_sets[i], dev_sets_move[i]],
                  ignore_index=True)
        for i in range(num_splits)
    ]

    return augmented_train_sets, dev_sets_new, dev_split_idx


def print_split_sizes(train_sets_by_annotator, dev_sets_by_annotator, num_splits):
    for i in range(num_splits):
        print("annotator {}: train set size {}, dev set size {}, total size {}".format(
            i, len(train_sets_by_annotator[i]), len(dev_sets_by_annotator[i]),
            len(train_sets_by_annotator[i]) + len(dev_sets_by_annotator[i])))


def get_annotator_splits(all_data, dev_annotator_list, num_splits, multi_mode):
    if multi_mode:
        dev_sets_by_annotator = [
            all_data.loc[all_data["turkIdAnonymized"].isin(dev_annotator_list[i])]
            for i in range(num_splits)
        ]
        train_sets_by_annotator = [
            all_data.loc[~all_data["turkIdAnonymized"].isin(dev_annotator_list[i])]
            for i in range(num_splits)
        ]
    else:
        dev_sets_by_annotator = [
            all_data.loc[all_data["turkIdAnonymized"] == dev_annotator_list[i]]
            for i in range(num_splits)
        ]
        train_sets_by_annotator = [
            all_data.loc[all_data["turkIdAnonymized"] != dev_annotator_list[i]]
            for i in range(num_splits)
        ]

    return train_sets_by_annotator, dev_sets_by_annotator


def get_random_splits(all_data, train_sets_by_annotator, dev_sets_by_annotator, num_splits):
    shuffled_data_frames = [
        all_data.sample(frac=1.0)
        for _ in range(num_splits)
    ]

    dev_sets_random = [
        shuffled_data_frames[i][0:dev_sets_by_annotator[i].shape[0]]
        for i in range(num_splits)
    ]
    train_sets_random = [
        shuffled_data_frames[i][dev_sets_by_annotator[i].shape[0]:
                                dev_sets_by_annotator[i].shape[0] + train_sets_by_annotator[i].shape[0]]
        for i in range(num_splits)
    ]

    return train_sets_random, dev_sets_random


def write_split_files(train_sets_by_annotator, dev_sets_by_annotator,
                      train_sets_random, dev_sets_random, num_splits,
                      multi_mode, augment_ratio, take_number, only_random, only_annotator):
    type_add = "_multi" if multi_mode else ""
    for i in range(0, num_splits):
        augment = "_augment{}".format(augment_ratio) if augment_ratio > 0 else ""
        take = "_take{}".format(take_number) if take_number > 1 else ""
        properties = augment + take

        if not only_random:
            dump_to_json(train_sets_by_annotator[i], "annotator" + type_add, "train", i, properties)
            dump_to_json(dev_sets_by_annotator[i], "annotator" + type_add, "dev", i, properties)

        if not only_annotator:
            dump_to_json(train_sets_random[i], "rand" + type_add, "train", i, properties)
            dump_to_json(dev_sets_random[i], "rand" + type_add, "dev", i, properties)


def create_random_augmented_series(multi_mode, take_number):
    all_data = load_data()
    print_annotator_distribution(all_data)

    sorted_annotators = all_data.groupby(['turkIdAnonymized']).nunique().sort_values(
        'id', ascending=False)['id'].keys()

    if multi_mode:
        dev_annotator_list = [sorted_annotators[i:i + 5].values for i in range(20, 45, 5)]
        num_splits = 5
    else:
        dev_annotator_list = [sorted_annotators[i] for i in range(5)]
        num_splits = 5

    train_sets_by_annotator, dev_sets_by_annotator = get_annotator_splits(
        all_data, dev_annotator_list, num_splits, multi_mode)

    train_sets_random, dev_sets_random = get_random_splits(
        all_data, train_sets_by_annotator, dev_sets_by_annotator, num_splits)

    write_split_files([], [], train_sets_random, dev_sets_random, num_splits,
                      multi_mode, 0.0, take_number,
                      only_random=True, only_annotator=False)

    for augment_ratio in [0.1, 0.2, 0.3]:
        print("before augmentation.")
        print_split_sizes(train_sets_random, dev_sets_random, num_splits)

        augmented_train_sets_by_annotator, augmented_dev_sets_by_annotator, dev_split_idx = augment_data(
            train_sets_by_annotator, dev_sets_by_annotator, num_splits, augment_ratio,
            by_annotator=True)

        augmented_train_sets_random, augmented_dev_sets_random, _ = augment_data(
            train_sets_random, dev_sets_random, num_splits, augment_ratio,
            by_annotator=False, dev_split_idx=dev_split_idx)

        print("after augmentation.")
        print_split_sizes(augmented_train_sets_random, augmented_dev_sets_random,
                          num_splits)

        for i in range(num_splits):
            assert augmented_train_sets_by_annotator[i].shape == augmented_train_sets_random[i].shape
            assert augmented_dev_sets_by_annotator[i].shape == augmented_dev_sets_random[i].shape
            assert augmented_train_sets_random[i].shape == train_sets_random[i].shape
            print("split {}: train {}, dev {}".format(
                i, augmented_train_sets_random[i].shape, augmented_dev_sets_random[i].shape))

        write_split_files([], [], augmented_train_sets_random, augmented_dev_sets_random,
                          num_splits, multi_mode, augment_ratio, take_number,
                          only_random=True, only_annotator=False)


def create_data_splits(multi_mode=True, augment_ratio=0.0, take_number=1, only_random=False, only_annotator=False):
    all_data = load_data()
    print_annotator_distribution(all_data)

    sorted_annotators = all_data.groupby(['turkIdAnonymized']).nunique().sort_values(
        'id', ascending=False)['id'].keys()

    if multi_mode:
        dev_annotator_list = [sorted_annotators[i:i+5].values for i in range(20, 45, 5)]
        num_splits = 5
    else:
        dev_annotator_list = [sorted_annotators[i] for i in range(5)]
        num_splits = 5

    train_sets_by_annotator, dev_sets_by_annotator = get_annotator_splits(
        all_data, dev_annotator_list, num_splits, multi_mode)

    train_sets_random, dev_sets_random = get_random_splits(
        all_data, train_sets_by_annotator, dev_sets_by_annotator, num_splits)

    if augment_ratio > 0:
        print("before augmentation.")
        print_split_sizes(train_sets_by_annotator, dev_sets_by_annotator, num_splits)

        train_sets_by_annotator, dev_sets_by_annotator, dev_split_idx = augment_data(
            train_sets_by_annotator, dev_sets_by_annotator, num_splits, augment_ratio,
            by_annotator=True)

        train_sets_random, dev_sets_random, _ = augment_data(
            train_sets_random, dev_sets_random, num_splits, augment_ratio,
            by_annotator=False, dev_split_idx=dev_split_idx)

        print("after augmentation.")
        print_split_sizes(train_sets_by_annotator, dev_sets_by_annotator, num_splits)

    for i in range(num_splits):
        assert train_sets_random[i].shape == train_sets_by_annotator[i].shape
        assert dev_sets_random[i].shape == dev_sets_by_annotator[i].shape
        print("split {}: train {}, dev {}".format(i, train_sets_random[i].shape, dev_sets_random[i].shape))

    write_split_files(train_sets_by_annotator, dev_sets_by_annotator,
                      train_sets_random, dev_sets_random, num_splits,
                      multi_mode, augment_ratio, take_number, only_random, only_annotator)


def main(args):
    for i in range(args.repeat):
        if args.augment_random_series:
            create_random_augmented_series(args.multi_mode,
                                           args.take_number + i)
        else:
            create_data_splits(args.multi_mode,
                               args.augment_ratio,
                               args.take_number + i,
                               args.only_random,
                               args.only_annotator)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="OpenbookQA: create data splits")
    parser.add_argument('--augment_ratio', type=float, default=0.0,
                        help='fraction of annotator examples to augment the train set with.')
    parser.add_argument('--take_number', type=int, default=1,
                        help='the number of times the specified split is being generated.'
                             'if repeat>1 then this argument will be the starting index of the generated splits.')
    parser.add_argument('--repeat', type=int, default=1,
                        help='how many times to generate the specified split.')

    parser.add_argument('--multi_mode', action='store_true', default=False,
                        help='multi-annotator mode.')
    parser.add_argument('--augment_random_series', action='store_true', default=False,
                        help='a series of augmentation splits, starting from random splits '
                             'corresponding to annotator splits in size.')
    parser.add_argument('--only_random', action='store_true', default=False,
                        help='generate only random splits.')
    parser.add_argument('--only_annotator', action='store_true', default=False,
                        help='generate only annotator splits.')

    args = parser.parse_args()

    main(args)


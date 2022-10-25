import json
import os
import random


def get_entirety(path):
    with open(path) as f_in:
        data = json.load(f_in)
        return data


def output_subset(data, output_path, output_file, percentage=0.25):
    num = int(len(data)*percentage)
    print('output {} samples'.format(num))
    random.shuffle(data)
    subset = []
    for index in range(0, num):
        subset.append(data[index])
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with open(os.path.join(output_path, output_file), 'w', encoding='utf-8') as f_out:
        json.dump(subset, f_out, ensure_ascii=False)


def main():
    filename = 'train.json'
    data_path = '../data/partial_processed'
    percentage = 0.5
    output_path = '../data/{}-partial_processed'.format(percentage)
    data = get_entirety(os.path.join(data_path, filename))
    output_subset(data, output_path, filename, percentage)


if __name__ == '__main__':
    main()

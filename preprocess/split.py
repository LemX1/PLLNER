import json
import os


def split(data, set_name, set_len, output_path):
    start = 0
    for index in range(len(set_name)):
        print('开始输出' + set_name[index] + '的' + str(set_len[index]) + '条数据')
        with open(os.path.join(output_path, set_name[index]+'.json'), 'w', encoding='utf-8') as f_out:
            end = start + set_len[index]
            print(start, end-1)
            subset = data[start:end]
            start = end
            json.dump(subset, f_out, ensure_ascii=False)
        print('输出完毕', '\n')


def read_data(data_path, filename='total.json'):
    with open(os.path.join(data_path, filename)) as f_in:
        data = json.load(f_in)
    return data


def cal_split_len(data, ratios):
    total = len(data)
    print('读取到' + str(total) + '条数据', '\n')
    used = 0
    left = 1
    split_len = []
    for ratio in ratios:
        left -= ratio
        num = int(total * ratio)
        if used + num <= total:
            split_len.append(num)
            used += num
        else:
            split_len.append(total - used)
            used = total
            left = 0
    if left > 0:
        split_len.append(total - used)
    elif used < total:
        split_len[-1] += total - used
    return split_len


def main():
    split_way = ['dev', 'test']
    split_ratios = [0.5, 0.5]

    assert len(split_way) == len(split_ratios) or len(split_way) == len(split_ratios) + 1,\
        'the number of ratios is wrong'
    total = 0
    for ratio in split_ratios:
        total += ratio
    assert 0 <= total <= 1, 'the sum of ratios more than 1 or less than 0'

    root = os.path.abspath('..')
    data_path = os.path.join(root, 'data/dev+test_version/include_same')
    data = read_data(data_path)
    split_len = cal_split_len(data, split_ratios)
    split_len = split_len[0:len(split_way)]
    split(data, split_way, split_len, data_path)


if __name__ == '__main__':
    main()

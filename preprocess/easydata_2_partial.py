import json
import os
import sys
import re


def gather(data_dir):
    results = {}
    sub_dirs = os.listdir(data_dir)
    for sub_dir in sub_dirs:
        root = os.path.join(data_dir, sub_dir)
        labelers = os.listdir(root)
        for labeler in labelers:
            path = os.path.join(root, labeler)
            file_list = os.listdir(path)
            for file in file_list:
                with open(os.path.join(path, file)) as f_in:
                    data = json.load(f_in)
                    if 'records' not in data:
                        continue
                    if data['content'] not in results:
                        results.update({data['content']: [data['records']]})
                    else:
                        results[data['content']].append(data['records'])
    return results


def remove_blank(seq):
    return seq.replace(' ', '').replace('\t', '').replace('\n', '')\
        .replace('\u3000', '').replace('\u00A0', '').replace('\u0020', '').replace('\u2800', '').replace('\u2005', '')


def unify_single_blank(seq):
    return seq.replace('\t', ' ').replace('\n', ' ')\
        .replace('\u3000', ' ').replace('\u00A0', ' ').replace('\u0020', ' ').replace('\u2800', ' ').replace('\u2005', ' ')


def combine_same(data):
    results = {}
    restore = {}
    for sample in data:
        no_blank_sample = remove_blank(sample)
        if no_blank_sample not in restore:
            restore.update({no_blank_sample: sample})
            results.update({sample: data[sample]})
        else:
            results[restore[no_blank_sample]] += data[sample]
    return results


def output(data, output_dir, encoding='bio'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    f_less_than_3 = open(os.path.join(output_dir, 'less_than_3.json'), 'w', encoding='utf-8')
    f_partial = open(os.path.join(output_dir, 'train.json'), 'w', encoding='utf-8')
    less_than_3 = []
    partial = []
    for content in data:
        if len(data[content]) < 3:
            less_than_3.append({
                content: data[content]
            })
        else:
            tokens = []
            for token in content:
                tokens.append(token)
            labels = []
            for records in data[content]:
                label = ['O'] * len(tokens)
                for record in records:
                    tag = record['tag']
                    if encoding == 'bio':
                        current = record['offset'][0]
                        end = record['offset'][1]
                        label[current] = 'B-'+tag
                        current += 1
                        while current <= end:
                            label[current] = 'I-'+tag
                            current += 1
                    elif encoding == 'bioes':
                        current = record['offset'][0]
                        end = record['offset'][1]
                        if current == end:
                            label[current] = 'S-'+tag
                        else:
                            label[current] = 'B-'+tag
                            current += 1
                            while current < end:
                                label[current] = 'I-'+tag
                                current += 1
                            label[current] = 'E-'+tag
                    else:
                        print('{} is an unsupported encoding way'.format(encoding))
                labels.append(label)
            partial.append({
                'tokens': tokens,
                'labels': labels
            })
    json.dump(less_than_3, f_less_than_3, ensure_ascii=False)
    json.dump(partial, f_partial, ensure_ascii=False)


def process_re_special_token(seq):
    return seq.replace('(', '\(').replace(')', '\)')\
        .replace('[', '\[').replace(']', '\]')


def revise_offset(data):
    results = {}
    for content in data:
        results.update({content: []})
        for records in data[content]:
            for record in records:
                entity = content[record['offset'][0]: record['offset'][1] + 1]
                if len(entity) != len(record['span']):  # 长度不相等，严重偏移
                    print('severe deviation detected!')
                    print(entity)
                    print(record['span'])
                    sys.exit()
                entity = unify_single_blank(entity)     # 将已知可能出现的空格转化成' ',下同
                record['span'] = unify_single_blank(record['span'])
                if entity != record['span']:  # 统一空格后仍然不同
                    new_content = unify_single_blank(content)
                    assert len(new_content) == len(content), '转换前后长度变化！'
                    spans = [substr.span() for substr in re.finditer(process_re_special_token(record['span']), new_content)]
                    assert len(spans) != 0, '文本中没有找到目标实体, 目标实体:{}, 文本:{}'.format(record['span'], new_content)
                    min_gap = 999
                    most_possible_span = None
                    for span in spans:
                        gap = abs(span[0] - record['offset'][0])
                        if gap < min_gap:
                            min_gap = gap
                            most_possible_span = span
                    entity = new_content[most_possible_span[0]: most_possible_span[1]]
                    if entity == record['span']:
                        record['offset'][0] = most_possible_span[0]
                        record['offset'][1] = most_possible_span[1] - 1
                    else:
                        print('still false!')
                        print(entity)
                        print(record['span'])
                        sys.exit()
            results[content].append(records)
    return results


def main():
    data = gather('../data/raw_unrevised')
    print(len(data))
    data = combine_same(data)
    print(len(data))
    data = revise_offset(data)
    print(len(data))
    output_dir = '../data/partial_processed'
    output(data, output_dir, encoding='bio')


if __name__ == '__main__':
    main()

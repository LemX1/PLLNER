import json
import os
import sys


def output(different, same, target_dir, filename_d='train.json', filename_s='unified_label_train.json'):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    filename_d = os.path.join(target_dir, filename_d)
    filename_s = os.path.join(target_dir, filename_s)
    with open(filename_d, 'w', encoding='utf-8') as f_out:
        json.dump(different, f_out, ensure_ascii=False)
        print('输出{}条标注不一致数据'.format(len(different)))
    with open(filename_s, 'w', encoding='utf-8') as f_out:
        json.dump(same, f_out, ensure_ascii=False)
        print('输出{}条标注一致数据'.format(len(same)))


def extract_entity(labels, encoding='bio'):
    entities = {
        '时间': [],
        '人物': [],
        '地点': []
    }
    label_num = len(labels)
    index = 0
    if encoding == 'bio':
        while index < label_num:
            if labels[index].startswith('B'):
                entity_type = labels[index].split('-')[1]
                start_index = index
                end_index = index
                index += 1
                while index < label_num:
                    if labels[index].startswith('B'):
                        break
                    elif labels[index].startswith('O'):
                        index += 1
                    elif labels[index].startswith('I'):
                        current_entity_type = labels[index].split('-')[1]
                        index += 1
                        if entity_type == current_entity_type:
                            end_index += 1
                        else:
                            break
                entities[entity_type].append((start_index, end_index))
            else:
                index += 1
    elif encoding == 'bioes':
        while index < label_num:
            if labels[index].startswith('S'):
                entities[labels[index].split('-')[1]].append((index, index))
                index += 1
            elif labels[index].startswith('B'):
                entity_type = labels[index].split('-')[1]
                start_index = index
                index += 1
                while index < label_num:
                    if labels[index].startswith('I'):
                        index += 1
                        if entity_type != labels[index].split('-')[1]:
                            break
                    elif labels[index].startswith('E'):
                        end_index = index
                        index += 1
                        if entity_type == labels[index].split('-')[1]:
                            entities[entity_type].append((start_index, end_index))
                        else:
                            break
                    else:
                        break
            else:
                index += 1
    return entities


def divide(data):
    different = []
    similar = []
    same = []
    threshold = 0.4
    for sample in data:
        labeler_num = len(sample['labels'])
        labels = []
        for label in sample['labels']:
            entities = extract_entity(label)
            labels.append(entities)
        count = {}
        for entities in labels:
            for entity_type in entities:
                for entity in entities[entity_type]:
                    keyname = '{}-{}-{}'.format(entity_type, entity[0], entity[1])
                    if keyname not in count:
                        count.update({keyname: 1})
                    else:
                        count[keyname] += 1
        entity_num = len(count)
        if entity_num == 0:
            print('检测到无实体数据：')
            print(sample['tokens'])
            same.append(sample)
            continue
        same_count = 0.
        for entity in count:
            if labeler_num == count[entity]:
                same_count += 1.
        similarity = same_count / float(entity_num)
        if similarity == 1:
            same.append(sample)
        elif similarity > threshold:
            similar.append(sample)
        else:
            different.append(sample)
        if similarity > 1 or similarity < 0:
            print('error ', similarity)
            sys.exit()
    print('有{}条数据标注结果相似'.format(len(similar)))
    print('有{}条数据标注结果差别较大'.format(len(different)))
    print('有{}条数据标注结果一致'.format(len(same)))
    return different, similar, same


def main():
    source_dir = '/root/partial_label_learning/data/1-partial_processed'
    target_dir = '/root/partial_label_learning/data/1-partial_selected_processed_less'
    data = json.load(open(os.path.join(source_dir, 'train.json'), 'r', encoding='utf-8'))
    different, similar, same = divide(data)
    output(different, same, target_dir)


if __name__ == '__main__':
    main()

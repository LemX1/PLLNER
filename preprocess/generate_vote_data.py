import os
import json
import sys


def vote(data):
    result = []
    for sample in data:
        labeler_num = len(sample['labels'])
        threshold = int(labeler_num / 2)
        labels = []
        for token_id, token in enumerate(sample['tokens']):
            count = {}
            for labeler_id in range(labeler_num):
                label = sample['labels'][labeler_id][token_id]
                if label not in count:
                    count.update({label: 1})
                else:
                    count[label] += 1
            not_append = True
            last = ''
            for label in count:
                last = label
                if count[label] > threshold:
                    labels.append(label)
                    not_append = False
                    break
            if not_append:
                labels.append(last)
        assert len(labels) == len(sample['tokens']), 'vote前后标签数不一致！！'
        result.append({
            'tokens': sample['tokens'],
            'labels': labels
        })
    print("共输出{}条数据".format(len(result)))
    return result
                    

def main():
    source_dir = '/root/partial_label_learning/data/1-partial_divergent_processed'
    target_dir = '/root/partial_label_learning/data/dev+test_version'
    data = json.load(open(os.path.join(source_dir, 'unified_label_train.json')))
    voted_data = vote(data)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    with open(os.path.join(target_dir, 'train.json'), 'w', encoding='utf-8') as f_out:
        json.dump(voted_data, f_out, ensure_ascii=False)


if __name__ == '__main__':
    main()

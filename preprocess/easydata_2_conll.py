import json
import os
import sys


def easydata_2_conll(raw_data, output_path, encoding_type='bio', output_file='total.json'):
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    with open(os.path.join(output_path, output_file), 'w', encoding='utf-8') as f_out:
        results = []
        for raw in raw_data:
            if 'records' in raw:
                entities = raw['records']
            else:
                continue
            content = raw['content']
            labels = ['O'] * len(content)
            for entity in entities:
                entity_type = entity['tag']
                start = entity['offset'][0]
                end = entity['offset'][1]
                if encoding_type == 'bio':
                    labels[start] = 'B-' + entity_type
                    for index in range(start+1, end+1):
                        labels[index] = 'I-' + entity_type
                elif encoding_type == 'bioes':
                    if start == end:
                        labels[start] = 'S-' + entity_type
                    else:
                        labels[start] = 'B-' + entity_type
                        for index in range(start+1, end):
                            labels[index] = 'I-' + entity_type
                        labels[end] = 'E-' + entity_type
            tokens = []
            for token in content:
                tokens.append(token)
            result = {
                'tokens': tokens,
                'labels': labels
            }
            results.append(result)
        json.dump(results, f_out, ensure_ascii=False)


def gather_raw_data(raw_data_path):
    raw_data_list = os.listdir(raw_data_path)
    total = []
    for raw_data in raw_data_list:
        with open(os.path.join(raw_data_path, raw_data)) as f_in:
            part = json.load(f_in)
            total.extend(part)
    print('共输出{}条数据'.format(len(total)))
    return total
    
    
def main():
    root = os.path.abspath('..')
    raw_data_path = os.path.join(root, 'data/vote')
    raw_data = gather_raw_data(raw_data_path)
    output_path = os.path.join(root, 'data/vote_processed')
    easydata_2_conll(raw_data, output_path)
    

if __name__ == '__main__':
    main()

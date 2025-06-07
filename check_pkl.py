import pickle

def print_pkl_content(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        print('Keys:', data.keys())
        if 'annotations' in data:
            print('\nFirst annotation:', data['annotations'][0])
            print('\nAnnotation keys:', data['annotations'][0].keys())
        if 'split' in data:
            print('\nSplit keys:', data['split'].keys())
            first_split_key = list(data['split'].keys())[0]
            print(f'\nFirst few items in {first_split_key}:', data['split'][first_split_key][:5])

if __name__ == '__main__':
    print_pkl_content('data/pkl/train.pkl') 
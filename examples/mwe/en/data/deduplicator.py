dataset = set()
with open('annotated_flower_data_metaphoric.txt') as f:
    lst = f.readlines()
    dataset = set(lst)
    print(f'duplicates in the dataset = {len(lst) - len(dataset)}')

dataset = list(dataset)
with open('annotated_flower_data_metaphoric_deduplicated.txt', 'w', encoding='utf-8') as f:
    f.writelines(dataset)

dataset = set()
with open('invalid_names.txt') as f:
    lst = f.readlines()
    dataset = set(lst)
    print(f'duplicates in the dataset = {len(lst) - len(dataset)}')

dataset = list(dataset)
with open('invalid_names_deduplicated.txt', 'w', encoding='utf-8') as f:
    f.writelines(dataset)

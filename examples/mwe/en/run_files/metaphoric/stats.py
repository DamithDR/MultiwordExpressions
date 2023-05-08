

file ='examples/mwe/en/data/metaphoric/spanish/spanish_annotation_filtered.txt'

with open(file,'r')as f:
    lines = f.readlines()

metaphoric_count = 0
total_count = 0
for line in lines:
    split=line.split(',')
    if split[2].strip().lower().__eq__('metaphoric'):
        metaphoric_count+=1
    total_count+=1

print(f'metaphoric = {metaphoric_count}')
print(f'total = {total_count}')
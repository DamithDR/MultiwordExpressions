lines = []
with open("flower_dataset_all_final_backup.tsv", 'r', encoding='utf-8') as f:
    lines = f.readlines()

final_lines = []
counter = 0
for line in lines:
    if len(line.strip()) == 1:
        counter += 1
        line = '\n'

    final_lines.append(line)
print(counter)
with open("flower_dataset_all_final_cleaned.tsv", 'w', encoding='utf-8') as f:
    f.writelines(final_lines)


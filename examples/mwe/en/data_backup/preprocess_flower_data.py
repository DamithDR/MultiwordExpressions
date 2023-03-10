lines = []
processed_lines = []
prev_line = ''

# remove numerics and single letters A,B,C and sanitise copyright statement
with open('PLANT_DICTIONARY.txt', encoding='utf-8') as f:
    lines = f.readlines()
# remove unwanted lines
redundant_lines = ['', '\n']
for i in range(0, len(lines)):
    line = lines[i]
    if line.strip() == '(c) 2012 Dorling Kindersley. All Rights Reserved.':
        line = ''

    if (len(line.strip()) != 1) & (not line.strip().isnumeric()):
        processed_lines.append(line)
    if line == '\n':
        prev_line = line
    else:
        prev_line = line.strip()  # at the end
f.close()
with open('flower_dataset.txt', 'w', encoding='utf-8') as f:
    f.writelines(processed_lines)
f.close()

# remove unwanted empty lines
f_prev_line = ''
with open('flower_dataset.txt', encoding='utf-8') as f:
    final_lines = f.readlines()
temp_process = []
final_process = []
for i in range(0, len(final_lines)):
    f_line = final_lines[i]
    if (f_prev_line == f_line) | ((f_prev_line in redundant_lines) & (f_line in redundant_lines)):
        print('prev line ' + f_prev_line)
        print("none found " + str(i) + " " + f_line)
    else:
        temp_process.append(f_line)

    if line == '\n':
        f_prev_line = f_line
    else:
        f_prev_line = f_line.strip()  # at the end
f.close()

for ll in range(0, len(temp_process) - 1):
    sent = temp_process[ll]
    if ((ll - 1 >= 0) & (temp_process[ll - 1].isupper())) & (
            (ll + 1 < len(temp_process)) & (temp_process[ll + 1].isupper())):
        print("found")
    else:
        final_process.append(sent)

# contextual processing
context_words = {}

context_processed = []
for n in range(0, len(final_process)):
    consider = final_process[n]

    if consider.isupper():
        if (n - 1 >= 0) & (not final_process[n - 1].isupper()):
            context_words[consider[:1]] = consider.replace('\n', '').lower()
        elif n == 0:
            context_words[consider[:1]] = consider.replace('\n', '').lower()

    split = consider.split(' ')
    for tok_id in range(0, len(split)):
        tok = split[tok_id]
        if (tok[:2].isupper()) & (tok[1:2] == '.') & (len(tok) == 2):
            if context_words.keys().__contains__(tok[:1]):
                split[tok_id] = context_words[tok[:1]]
    context_processed.append(" ".join(split))

with open('flower_dataset.txt', 'w', encoding='utf-8') as f:
    f.writelines(context_processed)
f.close()

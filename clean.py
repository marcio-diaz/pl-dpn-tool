
filename = 'test'
f_read = open(filename + ".c", 'r')
new_file_lines = ["typedef int pthread_t;"]

for line in f_read:
    sline = line.lstrip(' ')
    if sline[0] == "#":
        continue
    if sline[0] == "/":
        continue
    new_file_lines.append(line)

f_write = open(filename + "_clean.c", 'w')
f_write.write(''.join(new_file_lines))

f_read.close()
f_write.close()

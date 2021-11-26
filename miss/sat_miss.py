import os

count_cls = open('count.txt').readlines()
count_dict = {}
for count_item in count_cls:
    cat, val = count_item[:-1].split(',')
    if int(val) == 0:
        count_dict[int(cat)] = -1
    else:
        count_dict[int(cat)] = int(val)

file_list = ['miss_in_rank%d.txt'%x for x in [1,2,3]]
for file_name in file_list:
    data = open(file_name).readlines()
    wrong_pred = []
    wrong_label = []
    for item in data:
        pred, label = item[:-1].split(', ')[1:]
        wrong_pred.append(int(pred))
        wrong_label.append(int(label))
    file_io = open('count_'+file_name, 'a')
    for cls_num in range(5419):
        file_io.write(str(cls_num) + ',' + str(1.0*wrong_pred.count(cls_num)/count_dict[cls_num]) + ',' + str(1.0*wrong_label.count(cls_num)/count_dict[cls_num]) + str(count_dict[cls_num]) + '\n')
    file_io.close()


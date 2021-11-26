import json

ensamble_dir = './dcl_0.7_ensamble'
pkl_list = os.listdir(ensamble_dir)

anno_path = './../../FGVG_product/anno/sample_submission.csv'
anno = open(anno_path).readlines()[1:]

names = [x[:-1].split(' ')[0] for x in anno]

gt_gather = []
for file_name in pkl_list:
    data = json.load(open(os.path.join(ensamble_dir, file_name), 'rb')
    name_list =  data['id']
    if '/' in name_list[0]:
        name_list = [x.split('/')[-1] for x in name_list]
    
    if not name_list[0] in names:
        raise Exception('name not match %s <-> %s'%(name_list[0], names[0])
    for seq_name in names:
        gt_gather.append(data['probs'][name_list.index(seq_name)])

    data['id'] = names
    data['probs'] = gt_gather
    renamed_file = 'reranked_%s'%file_name
    save_io = open(os.path.join(ensmable_dir, rename_file, 'wb')
    pickle.dump(data, save_io)
    print('resaved %s'%renamed_file)
    

    
    


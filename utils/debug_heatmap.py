import cv2
import os
import numpy as np


def heatmap_debug_plot(heatmap, img_names, prefix):
    data_root = '../dataset/CUB_200_2011/dataset/data/'
    save_folder = './vis_tmp_save'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    counter = 0
    for sub_hm, img_name in zip(heatmap, img_names):
        np_hm = sub_hm.detach().cpu().numpy()
        np_hm -= np.min(np_hm)
        np_hm /= np.max(np_hm)  
        np_hm = np.uint8(255 * np_hm)
        np_hm = cv2.applyColorMap(np_hm, cv2.COLORMAP_JET)
        re_hm = cv2.resize(np_hm, (300, 300))
        
        raw_img = cv2.imread(os.path.join(data_root, img_name))
        re_img = cv2.resize(raw_img, (300, 300))
    
        canvas = np.zeros((300, 610, 3))
        canvas[:, :300, :] = re_img
        canvas[:, 310:, :] = re_hm
        save_name = img_name.split('/')[-1][:-4]
        cv2.imwrite(os.path.join(save_folder, prefix + '_' + save_name + '_' + str(counter) + '_heatmap_cmp.png'), canvas)
        counter += 1
    


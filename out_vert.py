#typical run would be something like python out_vert.py -c richardson -t 10
#This will take the richardson "face only" for richardson_targ_10 and calculate
#the vertices and save them in the richardson/vertices folder. You can use this to
#repose the source footage.

import numpy as np
import os
from glob import glob
import scipy.io as sio
from skimage.io import imread, imsave
from skimage.transform import rescale, resize
from time import time
import argparse
import ast

from api import PRN

def out_vert(args):

    # ---- init PRN
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu # GPU number, -1 for CPU
    prn = PRN(is_dlib = args.isDlib)

    # ------------- load data
   # image_folder = args.inputDir
   # print (image_folder)
    #save_folder = args.outputDir
    base_dir = args.baseDir
    character = args.characterDir
    target_num = args.targNum

    #e.g. d:\characters\richardson\face\richardson_t10    
    image_folder = "%s\\%s\\face\\%s_t%s" % (base_dir, character, character, target_num)
    print (image_folder)

    #e.g. d:\characters\richardson\vertices\richardson_t10
    save_folder = "%s\\%s\\vertices\\%s_t%s" % (base_dir, character, character, target_num)
    print (save_folder)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    types = ('*.jpg', '*.png')
    image_path_list= []
    for files in types:
        image_path_list.extend(glob(os.path.join(image_folder, files)))
    total_num = len(image_path_list)
    print (total_num)
    
    for i, image_path in enumerate(image_path_list):
        name = image_path.strip().split('\\')[-1][:-4]
        print (image_path)
        print (name)

        # read image
        image = imread(image_path)
        [h, w, _] = image.shape

        # the core: regress position map
        if args.isDlib:
            max_size = max(image.shape[0], image.shape[1])
            if max_size> 1000:
                image = rescale(image, 1000./max_size)
                image = (image*255).astype(np.uint8)
            pos = prn.process(image) # use dlib to detect face
        else:
            if image.shape[1] == image.shape[2]:
                image = resize(image, (256,256))
                pos = prn.net_forward(image/255.) # input image has been cropped to 256x256
            else:
                box = np.array([0, image.shape[1]-1, 0, image.shape[0]-1]) # cropped with bounding box
                pos = prn.process(image, box)
        
        image = image/255.
        if pos is None:
            continue


        vertices = prn.get_vertices(pos)
        np.save("%s/%s" % (save_folder, name), vertices)
        save_vertices = vertices.copy()
        save_vertices[:,1] = h - 1 - save_vertices[:,1]



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network')

    parser.add_argument('-i', '--inputDir', default='in', type=str,
                        help='path to the input directory, where input images are stored.')
    parser.add_argument('-o', '--outputDir', default='out', type=str,
                        help='path to the output directory, where results(obj,txt files) will be stored.')
    parser.add_argument('--gpu', default='0', type=str,
                        help='set gpu id, -1 for CPU')
    parser.add_argument('--isDlib', default=True, type=ast.literal_eval,
                        help='whether to use dlib for detecting face, default is True, if False, the input image should be cropped in advance')
    #mt arguments
    parser.add_argument('-b', '--baseDir', default='D:\\characters', type=str,
                        help='path to the directory containing all the character folders')
    parser.add_argument('-c', '--characterDir', default='beaver', type=str,
                        help='which character')
    parser.add_argument('-t', '--targNum', default='00', type=str,
                        help='which target file')
    # parser.add_argument('--is3d', default=True, type=ast.literal_eval,
    #                     help='whether to output 3D face(.obj)')

   
    out_vert(parser.parse_args())

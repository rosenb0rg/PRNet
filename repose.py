### This is a modified version of the PRNet demo.py script that looks for a directory of vertice arrays
### to reorient the source model to math the pose of hte target image (vertices)

###    python repose.py -c raupach -C richardson -s 001 -n 10  
### would take richardsons 
### source dialogue 001 from the rauapch scene, and reposition the face
### to match the orientation of richardson target 10. 

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

from utils.estimate_pose import estimate_pose
from utils.rotate_vertices import frontalize
from utils.align_vertices import align
from utils.render_app import get_visibility, get_uv_mask, get_depth_image
from utils.write import write_obj, write_obj_with_texture

def main(args):

    # ---- init PRN
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu # GPU number, -1 for CPU
    prn = PRN(is_dlib = args.isDlib)

    # ------------- load data
    # image_folder = args.inputDir
    # save_folder = args.outputDir
    # vertices_dir = args.vertDir
    
    #i.e. d:\source
    base_dir = args.baseDir
    
    #i.e. d:\characters
    base_save_dir = args.baseSavedir
    
    #i.e. source\raupach
    scene = args.sceneDir
    
    #i.e source\raupach\richardson (the target character)
    character = args.characterDir

    #i.e. source\rauapch\richardson\richardson_001
    source_num = args.sourceNum

#    targ_character = args.targChar
    #i.e. richardson_targ_10
    targ_num = args.targNum
    
    # something like D:\source\raupach\richardson\raupach_richardson_001
    image_folder = "%s\\%s\\%s\\%s_%s_%s" % (base_dir, scene, character, scene, character, source_num)
    print (image_folder)

    #something like d:\character\richardson\vertices\richards_t10
    vertices_dir = "%s\\%s\\vertices\\%s_t%s" % (base_save_dir, character, character, targ_num)
    print (vertices_dir)

    #something like d:\character\raupach\src\align\raupach_richardson_t10_s001
    save_folder = "%s\\%s\\src\\align\\%s_%s_s%s_t%s" % (base_save_dir, character, scene, character, source_num, targ_num )
    print (save_folder)


    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    # image_path_list= []
    # for root, dirs, files in os.walk('%s' % image_folder):
    #     for file in files:
    #         if file.endswith('.jpg'):
    #             image_path_list.append(file)
    # print (image_path_list)


    types = ('*.jpg', '*.png')
    image_path_list= []
    for files in types:
        image_path_list.extend(glob(os.path.join(image_folder, files)))
    total_num = len(image_path_list)
    image_path_list=sorted(image_path_list)
    #print (image_path_list)

# #repeating the above logic for a vertices directory.
    types = ('*.npy', '*.jpg')
    vert_path_list= []
    for files in types:
        vert_path_list.extend(glob(os.path.join(vertices_dir, files)))
    total_num_vert = len(vert_path_list)
   # vert_path_list.reverse()
    vert_path_list=sorted(vert_path_list)
    #print (vert_path_list)
        
    for i, image_path in enumerate(image_path_list):
        name = image_path.strip().split('\\')[-1][:-4]

        print ("%s aligned with %s" % (image_path_list[i], vert_path_list[i]))

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
        #takes the nth file in the directory of the vertices to "frontalize" the source image. 
        can_vert = vert_path_list[i]
        print (can_vert)
        save_vertices = align(vertices, can_vert)
        save_vertices[:,1] = h - 1 - save_vertices[:,1]

        colors = prn.get_colors(image, vertices)

        if args.isTexture:
            texture = cv2.remap(image, pos[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
            if args.isMask:
                vertices_vis = get_visibility(vertices, prn.triangles, h, w)
                uv_mask = get_uv_mask(vertices_vis, prn.triangles, prn.uv_coords, h, w, prn.resolution_op)
                texture = texture*uv_mask[:,:,np.newaxis]
            write_obj_with_texture(os.path.join(save_folder, name + '.obj'), save_vertices, colors, prn.triangles, texture, prn.uv_coords/prn.resolution_op)#save 3d face with texture(can open with meshlab)
        else:
            write_obj(os.path.join(save_folder, name + '.obj'), save_vertices, colors, prn.triangles) #save 3d face(can open with meshlab)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network')

    parser.add_argument('-i', '--inputDir', default='in', type=str,
                        help='path to the input directory, where input images are stored.')
    parser.add_argument('-o', '--outputDir', default='out', type=str,
                        help='path to the output directory, where results(obj,txt files) will be stored.')
    parser.add_argument('-v', '--vertDir', default='verts', type=str,
                        help='path to the target vertices directory for mactching orientation')
    parser.add_argument('--gpu', default='0', type=str,
                        help='set gpu id, -1 for CPU')
    parser.add_argument('--isDlib', default=True, type=ast.literal_eval,
                        help='whether to use dlib for detecting face, default is True, if False, the input image should be cropped in advance')
    parser.add_argument('--is3d', default=True, type=ast.literal_eval,
                        help='whether to output 3D face(.obj)')
    parser.add_argument('--isFront', default=False, type=ast.literal_eval,
                        help='whether to frontalize vertices(mesh)')
    # update in 2017/4/25
    parser.add_argument('--isTexture', default=False, type=ast.literal_eval,
                        help='whether to save texture in obj file')
    #Add args for base file director, character name, target number, source number
    parser.add_argument('-b', '--baseDir', default='D:\\source', type=str,
                        help='path to the directory containing all the character folders')
    parser.add_argument('-B', '--baseSavedir', default='D:\\characters', type=str,
                        help='path to the directory containing all the character folders')
    parser.add_argument('-s', '--sceneDir', default='raupach', type=str,
                        help='the name of the character on the witness stand (scene name, i.e. raupach)')
    parser.add_argument('-c', '--characterDir', default='', type=str,
                        help='which character is talking in the scene (i.e. beaver in Raupach questioning)')
    parser.add_argument('-S', '--sourceNum', default='1', type=str,
                        help='which line from the scene (i.e. richardson_001')
    # parser.add_argument('-t', '--targChar', default='1', type=str,
    #                     help='which target file')
    parser.add_argument('-t', '--targNum', default='01', type=str,
                        help='the target shot for the talking character (i.e. richardson_targ_10')   
    main(parser.parse_args())

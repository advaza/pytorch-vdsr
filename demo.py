import argparse, os
import torch
from torch.autograd import Variable
from scipy.ndimage import imread
from PIL import Image
import numpy as np
import time, math
#import matplotlib.pyplot as plt

import os
import easyargs
import progressbar
import imageio
import glob
import cv2

parser = argparse.ArgumentParser(description="PyTorch VDSR Demo")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="model/model_epoch_50.pth", type=str, help="model path")
parser.add_argument("--image", default="butterfly_GT", type=str, help="image name")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument("--in_folder", default=None, type=str, help="input folder")
parser.add_argument("--output_dir", default=None, type=str, help="output folder")





def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)
    
def colorize(y, ycbcr): 
    img = np.zeros((y.shape[0], y.shape[1], 3), np.uint8)
    img[:,:,0] = y
    img[:,:,1] = ycbcr[:,:,1]
    img[:,:,2] = ycbcr[:,:,2]
    # img = Image.fromarray(img, "YCbCr").convert("RGB")
    return img


def upscale_function(image, opt):

    cuda = opt.cuda

    if cuda:
        #print("=> use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
                raise Exception("No GPU found or Wrong gpu id, please run without --cuda")


    model = torch.load(opt.model, map_location=lambda storage, loc: storage)["model"]

    # im_gt_ycbcr = imread("Set5/" + opt.image + ".bmp", mode="YCbCr")
    # im_b_ycbcr = imread("Set5/"+ opt.image + "_scale_"+ str(opt.scale) + ".bmp", mode="YCbCr")

    # im_gt_y = im_gt_ycbcr[:,:,0].astype(float)
    im_b_y = image[:,:,0].astype(float)

    # psnr_bicubic = PSNR(im_gt_y, im_b_y,shave_border=opt.scale)

    im_input = im_b_y/255.

    im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[0], im_input.shape[1])

    if cuda:
        model = model.cuda()
        im_input = im_input.cuda()
    else:
        model = model.cpu()

    start_time = time.time()
    out = model(im_input)
    elapsed_time = time.time() - start_time

    out = out.cpu()

    im_h_y = out.data[0].numpy().astype(np.float32)

    im_h_y = im_h_y * 255.
    im_h_y[im_h_y < 0] = 0
    im_h_y[im_h_y > 255.] = 255.

    # psnr_predicted = PSNR(im_gt_y, im_h_y[0,:,:], shave_border=opt.scale)

    im_h = colorize(im_h_y[0,:,:], image)
    # im_gt = Image.fromarray(im_gt_ycbcr, "YCbCr").convert("RGB")
    #im_b = Image.fromarray(im_b_ycbcr, "YCbCr").convert("RGB")

    #print("Scale=",opt.scale)
    # print("PSNR_predicted=", psnr_predicted)
    # print("PSNR_bicubic=", psnr_bicubic)
    #print("It takes {}s for processing".format(elapsed_time))
    return im_h
    # fig = plt.figure()
    # ax = plt.subplot("131")
    # ax.imshow(im_gt)
    # ax.set_title("GT")
    #
    # ax = plt.subplot("132")
    # ax.imshow(im_b)
    # ax.set_title("Input(bicubic)")
    #
    # ax = plt.subplot("133")
    # ax.imshow(im_h)
    # ax.set_title("Output(vdsr)")
    # plt.show()


def folders_in(directory, recursive=True):

    all_folders = [directory]
    # silly hack to handle file streams which respond only after query
    for root, dirnames, filenames in os.walk(directory):
        if not recursive:
            return dirnames
        all_folders.extend(dirnames)
    return all_folders


def filter_files(filenames, extensions):
    return [name for name in filenames
            if os.path.splitext(name)[-1].lower() in extensions and '_VDSR' not in name]


def files_in(directory, extensions, recursive=False):
    all_files = []
    for root, dirnames, filenames in os.walk(directory):
        curr_files = filter_files(filenames, extensions)
        if curr_files:
            curr_files = [os.path.join(directory, filename) for filename in curr_files]
            all_files.extend(curr_files)
        if not recursive:
            return all_files
    return all_files


def process_image(image, opt):

    image = image.astype(np.float32)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)

    output_image = upscale_function(image, opt)
    output_image = (output_image).astype(np.uint8)
    output_image = cv2.cvtColor(output_image, cv2.COLOR_YCR_CB2BGR)
    return output_image


def process_out_file_path(file_name, output_dir):

    if output_dir is None:
        output_dir = os.path.abspath(os.path.dirname(file_name))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    basename = os.path.basename(file_name)
    extension = basename.split('.')[-1]
    out_name = basename[:-len(extension)-1] + '_VDSR' + '.' + extension

    return os.path.join(output_dir, out_name)


def main():
    """
    Calculate histogram transfer from reference image to a given video
    :param in_folder: Input folder of folders with video files
    :return:
    """
    opt = parser.parse_args()
    folders = folders_in(opt.in_folder, recursive=True)
    for folder in folders:
        video_files = files_in(folder, extensions=['.mp4'])
        image_files = files_in(folder, extensions=['jpg', 'JPG', 'png', 'jpeg', 'JPEG'])

        reuse=False
        if image_files:
            for image_file in image_files:

                out_file = process_out_file_path(image_file, opt.output_dir)
                image = cv2.imread(image_file)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                out_image = process_image(image, opt)
                out_image = cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(out_file, out_image)
                if not reuse:
                    reuse = True

        if video_files:
            for video_file in video_files:

                video_reader = imageio.get_reader(video_file)
                out_video = process_out_file_path(video_file, opt.output_dir)
                writer = imageio.get_writer(out_video, fps=video_reader.get_meta_data()['fps'])
                print('Working on %s' % out_video)

                bar = progressbar.ProgressBar()
                for frame in bar(video_reader):

                    writer.append_data(process_image(frame, opt))
                    if not reuse:
                        reuse = True
                writer.close()


if __name__ == '__main__':
    main()
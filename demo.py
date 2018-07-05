import easyargs
import progressbar
import imageio
import os
import glob
import torch
import cv2

from torch.autograd import Variable
import numpy as np

IMAGE_FORMATS = ['jpg', 'png', 'jpeg']
VIDEO_FORMATS = ['mp4']


def colorize(y, ycbcr): 
    img = np.zeros((y.shape[0], y.shape[1], 3), np.uint8)
    img[:,:,0] = y
    img[:,:,1] = ycbcr[:,:,1]
    img[:,:,2] = ycbcr[:,:,2]
    img = np.ndarray.astype(img, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_YCR_CB2RGB)
    return img


def process_image(im_input, model, cuda):

    im_b_ycbcr = cv2.cvtColor(im_input, cv2.COLOR_RGB2YCR_CB)
    im_b_y = im_b_ycbcr[:,:,0].astype(float)
    im_input = im_b_y/255.

    im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[0], im_input.shape[1])
    if cuda:
        im_input = im_input.cuda()

    out = model(im_input)
    out = out.cpu()

    im_h_y = out.data[0].numpy().astype(np.float32)

    im_h_y = im_h_y * 255.
    im_h_y[im_h_y < 0] = 0
    im_h_y[im_h_y > 255.] = 255.

    im_h = colorize(im_h_y[0,:,:], im_b_ycbcr)

    return im_h


def files_in(input_path, extensions):

    all_files = []
    for suffix in extensions:
        all_files.append(glob.glob(os.path.join(input_path, '*.' + suffix.lower())))
    return all_files if all_files else None


def process_out_file_path(file_name, output_dir):

    if output_dir is None:
        output_dir = os.path.abspath(os.path.dirname(file_name))

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    basename = os.path.basename(file_name)
    extension = basename.split('.')[-1]
    out_name = basename[:-len(extension)-1] + '_VDSR' + '.' + extension

    return os.path.join(output_dir, out_name)

@easyargs
def main(input_path=None, output_dir=None, model_checkpoint='model/model_epoch_50.pth', cuda=False,
         gpus="0"):

    if cuda:
        print("=> use gpu id: '{}'".format(gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    bar = progressbar.ProgressBar()

    model = torch.load(model_checkpoint, map_location=lambda storage, loc: storage)["model"]
    if cuda:
        model = model.cuda()
    else:
        model = model.cpu()

    image_files, video_files = None, None
    if os.path.isfile(input_path):
        extension = os.path.basename(input_path).split('.')[-1]
        if extension in IMAGE_FORMATS:
            image_files = [input_path]
        elif extension in VIDEO_FORMATS:
            video_files = [input_path]
        else:
            raise ValueError(
                'Input path should be either a folder or a file of image/video format.')
    else:
        video_files = files_in(input_path, extensions=VIDEO_FORMATS)
        image_files = files_in(input_path, extensions=IMAGE_FORMATS)

    if image_files:
        for image_file in bar(image_files):

            out_file = process_out_file_path(image_file, output_dir)
            image = cv2.imread(image_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            out_image = process_image(image, model, cuda)
            out_image = cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(out_file, out_image)

    if video_files:
        try:
            for video_file in video_files:
                video_reader = imageio.get_reader(video_file)
                out_video = process_out_file_path(video_file, output_dir)
                writer = imageio.get_writer(out_video, fps=video_reader.get_meta_data()['fps'])
                print('Working on %s' % out_video)

                for frame in bar(video_reader):

                    writer.append_data(process_image(frame, model, cuda))
                writer.close()
        except RuntimeError:  # if there is a problem with the video file.
            pass


if __name__ == '__main__':
    main()

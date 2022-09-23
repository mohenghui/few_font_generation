import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from real_esrgan.inference_realesrgan_my import FontResolution
from miximg import rename_inference
if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options #创建模型
    model.setup(opt)               # regular setup: load and print networks; create schedulers #创建学习率
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default #从0开始
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    if opt.eval:
        model.eval() #不更新权重
    for i, data in enumerate(dataset):
        #if i >= opt.num_test:  # only apply our model to opt.num_test images.
        #    break
        model.set_input(data)  # unpack data from data loader 打开数据
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 1000 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    input_dir=webpage.get_image_dir()
    print("生成完成")

    # input_dir="results/B端24/test_unknown_style_latest/images"
    FontResolution(input=input_dir,model='RealESRGAN_x4plus',scale=8,enhance=True,half=True).main()
    print("超分完成")
    for i in os.listdir(input_dir):
        rename_inference(opt.font_list,os.path.join(input_dir,i))
    print("改名完成")
    # webpage.save()  # save the HTML

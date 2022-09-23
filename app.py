from fileinput import filename
import os
from posixpath import dirname
from random import shuffle
from tkinter import font
from options.test_options_app import TestOptionsApp
from options.train_options_app import TrainOptionsApp
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import zipfile
import shutil
from real_esrgan.inference_realesrgan_my import FontResolution
from miximg import  judge, rename, rename_inference,makedir,resize
from flask import Flask, render_template, Response, request, redirect, url_for,send_file
from datasets.font2image_app import ttf2img
# python test.py --dataroot ./datasets/font  --model font_translator_gan  --eval --name MLANH --no_dropout 
import sys
from ocr.test import ocr
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import time
root_path=sys.path[0]
save_dir=os.path.join(root_path,"datasets/finetune/test_unknown_style/reference")
train_dir=os.path.join(root_path,"datasets/finetune/train/")

def inference():
    global input_dir
    os.chdir(sys.path[0])
    opt = TestOptionsApp().parse()  # get test options
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    # print(opt)
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options #创建模型
    model.setup(opt)               # regular setup: load and print networks; create schedulers #创建学习率
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default #从0开始
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    input_dir=webpage.get_image_dir()
    if not os.path.exists(input_dir):
        os.mkdir(input_dir)
    else:
        shutil.rmtree(input_dir) 
        os.mkdir(input_dir)
    # test with eval mode. This only affects layers like batchnorm and dropout.
    if opt.eval:
        model.eval() #不更新权重
    for i, data in enumerate(dataset):
        model.set_input(data)  # unpack data from data loader 打开数据
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 1000 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    
    print("生成完成")

    # input_dir="results/B端24/test_unknown_style_latest/images"
    FontResolution(input=input_dir,model='RealESRGAN_x4plus',scale=8,enhance=True,half=True).main()
    print("超分完成")
    for i in os.listdir(input_dir):
        rename_inference(opt.font_list,os.path.join(input_dir,i))
    print("改名完成")

def font_ttf(file_path):
    ttf2img(file_path,save_dir)
    if not os.listdir( os.path.join(save_dir,os.listdir(save_dir)[0])):
        return redirect(url_for('error'))
    inference()
def font_zip(file_path):
    global ERROR_FLAG
    f = zipfile.ZipFile(file_path,'r') # 压缩文件位置
    makedir(save_dir)
    for file in f.namelist():
        try:
            new_file_name = file.encode('cp437').decode('gbk')
        except:
            new_file_name = file.encode('utf-8').decode('utf-8')
        f.extract(file,save_dir)               # 解压位置
        os.rename(os.path.join(save_dir,file),os.path.join(save_dir,new_file_name))
        # judge()
    f.close()
    
    # dir_list=[]
    dir_list=[dirnames[0] for  _, dirnames, _ in os.walk(save_dir)if dirnames]
    for  dirpath, _, filenames in os.walk(save_dir):
        if not dir_list:
            new_dir=os.path.join(dirpath,"myfont")
            makedir(new_dir)
            for file in filenames:
                old_new_path=os.path.join(dirpath,file)
                new_file_path=os.path.join(new_dir,file)
                os.rename(old_new_path, new_file_path)
        elif len(dir_list)>1 and filenames:
            new_dir=os.path.join(save_dir,dir_list[0])
            for file in filenames:
                old_new_path=os.path.join(dirpath,file)
                new_file_path=os.path.join(new_dir,file)
                os.rename(old_new_path,new_file_path)
    if len(dir_list)>1:shutil.rmtree(os.path.join(save_dir,dir_list[0],dir_list[1]))
    if not os.listdir(os.path.join(save_dir,dir_list[0])) or not resize(new_dir,new_dir,64) :
        ERROR_FLAG=True
        return
    # ocr(root_path,new_dir) #ocr对压缩内容打上标签
    if request.form['class1']=="yes":finetune()
    inference()

#微调
def finetune():
    opt = TrainOptionsApp().parse()   # get training options #获得配置参数
    MAKE_DIR_FLAG=False
    for  dirpath, _, filenames in os.walk(save_dir):
        if filenames:
            # for dir in dirpath:
            for filename in filenames:
                old_name=os.path.join(dirpath,filename)
                new_name_reference=os.path.join(train_dir,"reference","myfont",filename)
                new_name_target=os.path.join(train_dir,"target","myfont",filename)
                if not MAKE_DIR_FLAG:
                    makedir(new_name_reference,True)
                    makedir(new_name_target,True)
                    MAKE_DIR_FLAG=True
                shutil.copy(old_name, new_name_reference)
                shutil.copy(old_name,new_name_target)
    old_model_dir=os.path.join(opt.checkpoints_dir,opt.name)
    new_model_dir=os.path.join(opt.checkpoints_dir,"finetune")
    opt.name="finetune"
    makedir(new_model_dir)
    for file in os.listdir(old_model_dir):
        shutil.copy(os.path.join(old_model_dir,file), os.path.join(new_model_dir,file))
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options #构建数据集
    dataset_size = len(dataset)    # get the number of images in the dataset. #数据集的数量
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing #输入数据
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights   
            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.


app = Flask(__name__)
@app.route('/')
def index():
    """Video streaming home page."""
    return render_template("index.html")

@app.route('/error')
def error():
    """Video streaming home page."""
    return render_template("error.html")


@app.route('/font_upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        # font_class=request.form['class']
        # print(font_class)
        basepath = os.path.dirname(__file__)  # 当前文件所在路径
        upload_path = os.path.join(root_path,'static/uploads')
        if not os.path.exists(upload_path):
            os.mkdir(upload_path)
        else:
            shutil.rmtree(upload_path) 
            os.mkdir(upload_path)
        upload_file_path = os.path.join(basepath, upload_path, (f.filename))  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
        f.save(upload_file_path)
        global NAME, GENERATE_FLAG,CAMERA_FLAG,ERROR_FLAG
        file_name,file_tail=os.path.splitext(f.filename)
        NAME = upload_file_path
        ERROR_FLAG=False
        file_tail_list=[".ttf",".otf"]
        if os.path.exists(save_dir):shutil.rmtree(save_dir) 
        if file_tail==".zip":
            font_zip(upload_file_path)
        elif file_tail.lower() in file_tail_list:
            font_ttf(upload_file_path)
        else:
            return redirect(url_for('error'))
        print(ERROR_FLAG)
        if ERROR_FLAG==False:
            GENERATE_FLAG = True
            return redirect(url_for('download_all'))
        else:return redirect(url_for('error'))
    else:return redirect(url_for('index'))




@app.route('/download')
def download_all():
    zipf = zipfile.ZipFile('Font.zip','w', zipfile.ZIP_DEFLATED)
    filepath = os.path.join(root_path,input_dir)
    app.logger.info(filepath)
    for root,dirs, files in os.walk(filepath):
        for file in files:
            os.chdir(root)  # 定位到文件夹
            zipf.write(file)
    zipf.close()
    return send_file('Font.zip',
            mimetype = 'zip',
            download_name= 'Font.zip',
            as_attachment = True)


if __name__ == '__main__':
    app.run(host='192.168.106.1', threaded=True, port=8080)
    # @app.route('/font_generstation')
    # def font_feed():
    
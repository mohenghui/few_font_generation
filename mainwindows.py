from email.errors import MisplacedEnvelopeHeaderDefect
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import cv2
import glob
import random
import numpy as np
from tkinter import *
from tkinter import filedialog
from xml.etree import ElementTree as ET
import tkinter.messagebox 
import natsort
class FontGan():
    def __init__(self) :
        self.img_dir = r""
        self.font_list_dir = r""
        self.save_dir = r""
        self.annotation_dir = r""
        self.detect_class = r""
        self.rename_dir = r""
        self.first_index = None
        self.file_tail = r""
        self.new_name = r""
        self.handle_flag = False
        self.predefined_classes = []
        self.objectList = []
        self.mix_flag = False
        self.rename_flag = False
        self.root_window = None

    def testModel(self):
            opt = TestOptions().parse()  # get test options
            # opt.dataroot=os.path.dirname(self.img_dir)
            opt.dataroot='./datasets/font'
            opt.model="font_translator_gan "
            opt.eval=True
            # opt.name="MLANH"
            opt.no_dropout=True
            # opt.results_dir=self.save_dir
            opt.results_dir="./results"
            # opt.phase=os.path.basename(self.img_dir)
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
                model.eval()
            for i, data in enumerate(dataset):
                #if i >= opt.num_test:  # only apply our model to opt.num_test images.
                #    break
                model.set_input(data)  # unpack data from data loader
                model.test()           # run inference
                visuals = model.get_current_visuals()  # get image results
                img_path = model.get_image_paths()     # get image paths
                if i % 1000 == 0:  # save images to an HTML file
                    print('processing (%04d)-th image... %s' % (i, img_path))
                save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
            webpage.save()  # save the HTML


    def client(self):
        def creatWindow():
            self.root_window.destroy()
            window()

        def judge(str):
            if (str):
                text = "你已选择" + str
            else:
                text = "你还未选择文件夹，请选择"
            return text

        def test01():
            self.img_dir = r""
            self.img_dir += filedialog.askdirectory()
            creatWindow()

        def test02():
            self.font_list_dir = r""
            self.font_list_dir += filedialog.askopenfilename()
            creatWindow()

        def test03():
            self.save_dir = r""
            self.save_dir += filedialog.askdirectory()
            creatWindow()
        def test06():
            # messagebox.showwarning('警告','明日有大雨')
            if not (self.save_dir and self.font_list_dir and self.img_dir):
                tkinter.messagebox.showwarning('警告','文件路径不能为空，请自行检查')
            else:
                self.mix_flag = True
                self.testModel()
            creatWindow()
            
        
        def window():
            self.root_window = Tk()
            self.root_window.title("")
            screenWidth = self.root_window.winfo_screenwidth()  # 获取显示区域的宽度
            screenHeight = self.root_window.winfo_screenheight()  # 获取显示区域的高度
            tk_width = 500  # 设定窗口宽度
            tk_height = 400  # 设定窗口高度
            tk_left = int((screenWidth - tk_width) / 2)
            tk_top = int((screenHeight - tk_width) / 2)
            self.root_window.geometry('%dx%d+%d+%d' % (tk_width, tk_height, tk_left, tk_top))
            self.root_window.minsize(tk_width, tk_height)  # 最小尺寸
            self.root_window.maxsize(tk_width, tk_height)  # 最大尺寸
            self.root_window.resizable(width=False, height=False)
            btn_1 = Button(self.root_window, text='请选择你要参考的风格图片文件夹', command=test01,
                           height=0)
            btn_1.place(x=175, y=40, anchor='w')

            text = judge(self.img_dir)
            text_label = Label(self.root_window, text=text)
            text_label.place(x=166, y=70, anchor='w')

            btn_2 = Button(self.root_window, text='请选择生成字体的清单(.txt)', command=test02,
                           height=0)
            btn_2.place(x=175, y=100, anchor='w')

            text = judge(self.font_list_dir)
            text_label = Label(self.root_window, text=text)
            text_label.place(x=166, y=130, anchor='w')

            btn_3 = Button(self.root_window, text='请选择要保存生成图片的文件夹', command=test03,
                           height=0)
            btn_3.place(x=175, y=160, anchor='w')
            text = judge(self.save_dir)
            text_label = Label(self.root_window, text=text)
            text_label.place(x=166, y=190, anchor='w')

            btn_6 = Button(self.root_window, text='开始生成', command=test06,
                           height=0)
            btn_6.place(x=175, y=340, anchor='w')

            if (self.mix_flag):
                text = "生成完成"
            else:
                text = "等待生成"
            text_label = Label(self.root_window, text=text)
            text_label.place(x=166, y=370, anchor='w')

            

            self.root_window.mainloop()

        window()

if __name__ == '__main__':
    FontGan().client()

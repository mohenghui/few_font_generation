from email.policy import default
from .base_options_app import BaseOptionsApp
from flask import  request

class TestOptionsApp(BaseOptionsApp):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """
    # python test.py --dataroot ./datasets/test_font  --model font_translator_gan --dataset_mode font_test   --eval --name B端24  --no_dropout

    # python test.py --dataroot ./datasets/font  --model font_translator_gan  --eval --name MLANH --no_dropout 
    def initialize(self, parser):
        parser = BaseOptionsApp.initialize(self, parser)  # define shared options
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test_unknown_style', help='train, test_unknown_style, test_unknown_content, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true',default="True", help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
        # rewrite devalue values
        if request.form['class']=="1500":
            parser.add_argument('--font_list', type=str, default='./datasets/sortfont1500.txt', help='choose generation font list')
        elif request.form['class']=="6864":
            parser.add_argument('--font_list', type=str, default='./datasets/sortfont.txt', help='choose generation font list')
        elif request.form['class']=="20902":
            parser.add_argument('--font_list', type=str, default='./datasets/sortfont20902.txt', help='choose generation font list')
        
        parser.set_defaults(model='font_translator_gan')
        if request.form['class1']=="yes":
            parser.set_defaults(name='finetune')
        else:
            parser.set_defaults(name='B端24')
        parser.add_argument('--evaluate_mode', type=str, default='content')
        
        self.isTrain = False
        return parser
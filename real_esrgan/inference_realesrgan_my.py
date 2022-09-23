import argparse
import cv2
import glob
import os
import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relativec
from basicsr.archs.rrdbnet_arch import RRDBNet

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

class FontResolution():
    def __init__(self,input='./inputs',output=None,model='RealESRGAN_x4plus',scale=8,enhance=True,half=True) :
        self.input=input
        self.output=self.input if not output else output
        self.model_name=model
        self.outscale=scale
        self.face_enhance=enhance
        self.fp32=half
        self.suffix='out'
        self.tile=0
        self.tile_pad=10
        self.pre_pad=0
        self.alpha_upsampler='realesrgan'
        self.ext='auto'
        self.gpu_id=0
        self.result=[]
    def findfiles(self,path):
    # 首先遍历当前目录所有文件及文件夹
        file_list = os.listdir(path)
        # 循环判断每个元素是否是文件夹还是文件，是文件夹的话，递归
        for file in file_list:
            # 利用os.path.join()方法取得路径全名，并存入cur_path变量，否则每次只能遍历一层目录
            cur_path = os.path.join(path, file)
            # 判断是否是文件夹
            if os.path.isdir(cur_path):
                self.findfiles(cur_path)
            # print(path,file)
            self.result.append(os.path.join(path,file))
    def main(self):

        # determine models according to model names
        self.model_name = self.model_name.split('.')[0]
        if self.model_name in ['RealESRGAN_x4plus', 'RealESRNet_x4plus']:  # x4 RRDBNet model
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
        elif self.model_name in ['RealESRGAN_x4plus_anime_6B']:  # x4 RRDBNet model with 6 blocks
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            netscale = 4
        elif self.model_name in ['RealESRGAN_x2plus']:  # x2 RRDBNet model
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            netscale = 2
        elif self.model_name in ['realesr-animevideov3']:  # x4 VGG-style model (XS size)
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
            netscale = 4

        # determine model paths
        model_path = os.path.join('real_esrgan/experiments/pretrained_models', self.model_name + '.pth')
        if not os.path.isfile(model_path):
            model_path = os.path.join('real_esrgan/realesrgan/weights', self.model_name + '.pth')
        if not os.path.isfile(model_path):
            raise ValueError(f'Model {self.model_name} does not exist.')

        # restorer
        upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            model=model,
            tile=self.tile,
            tile_pad=self.tile_pad,
            pre_pad=self.pre_pad,
            half=not self.fp32,
            gpu_id=self.gpu_id)

        if self.face_enhance:  # Use GFPGAN for face enhancement
            from gfpgan import GFPGANer
            face_enhancer = GFPGANer(
                model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
                upscale=self.outscale,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=upsampler)
        os.makedirs(self.output, exist_ok=True)

        if os.path.isfile(self.input):
            paths = [self.input]
        else:
            # paths = sorted(glob.glob(os.path.join(self.input, '*')))
            self.findfiles(self.input)
        # print(self.result)
        for idx, path in enumerate(self.result):
            print(path)
            imgname, extension = os.path.splitext(os.path.basename(path))
            # print('Testing', idx, imgname)
            if not os.path.isfile(path):continue
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if len(img.shape) == 3 and img.shape[2] == 4:
                img_mode = 'RGBA'
            else:
                img_mode = None

            try:
                if self.face_enhance:
                    _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
                else:
                    output, _ = upsampler.enhance(img, outscale=self.outscale)
            except RuntimeError as error:
                print('Error', error)
                print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
            else:
                if self.ext == 'auto':
                    extension = extension[1:]
                else:
                    extension = self.ext
                if img_mode == 'RGBA':  # RGBA images should be saved in png format
                    extension = 'png'
                if self.suffix == '':
                    # save_path = os.path.join(self.output, f'{imgname}.{extension}')
                    save_path = os.path.join(os.path.dirname(path), f'{imgname}.{extension}')
                else:
                    # save_path = os.path.join(self.output, f'{imgname}_{self.suffix}.{extension}')
                    save_path = os.path.join(os.path.dirname(path), f'{imgname}.{extension}')
                # print(save_path)
                cv2.imwrite(save_path, output)


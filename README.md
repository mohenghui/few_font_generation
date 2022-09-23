# few_font_generation   
[模型与数据集链接: ](https://pan.baidu.com/s/19Q8KY5f5aqWxLKsZzIlzZA) 提取码: jnp4    
**解压后放在文件夹中即可**   
# flask客户端   
python app.py   

## Requirements

* Linux
* CPU or NVIDIA GPU + CUDA CuDNN
* Python 3
* torch>=0.4.1
* torchvision>=0.2.1
* dominate>=2.3.1
* visdom>=0.1.8.3

- Train the model
```bash
bash ./train.sh
```
- Test
```bash
bash ./test.sh
```

- Evaluate
```bash
bash ./evaluate.sh
```

- scripts.sh integrate train.sh, test.sh, and evaluate.sh
```bash
bash ./scripts.sh
```

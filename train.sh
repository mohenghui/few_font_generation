set -ex
python train.py --dataroot ./datasets/font --model font_translator_gan --name test_new_dataset --no_dropout
python train.py --dataroot ./datasets/test_font --model font_translator_gan --name test_new_dataset --no_dropout  --n_epochs 15 --n_epochs_decay 15 --continue_train

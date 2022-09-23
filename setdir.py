import os

s05=os.listdir( './datasets/ttf2img/img_save')
s06=os.listdir( './datasets/ttf2img/img_save_test')

print('集合比较')
ans=[]
# print(s05)
# print(s06)
for i in s06:
    if i not in s05:ans.append(i)
# print(ans)
ttf_list=os.listdir('./datasets/ttf2img/ttf_save')
ttf_ans=[]
for i in ttf_list:
    ttf_ans.append(i.split('.')[0])
# print(ttf_ans)
for i in ttf_list:
    file_name=i.split('.')[0]
    file_end=i.split('.')[1]
    if file_name not in ans:
        os.remove('./datasets/ttf2img/ttf_save/'+i)

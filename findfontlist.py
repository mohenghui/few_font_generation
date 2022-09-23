import os
path=""
#判断路径是否存在
if os.path.exists(path):
    
    #获取该目录下所有文件，存入列表中
    fileList=os.listdir(path)
 
    n=0
    for i in fileList:
        
        #设置旧文件名（就是路径+文件名）
        oldname=path+ os.sep + fileList[n]   # os.sep添加系统分隔符
        #判断当前是否是文件
        if os.path.isfile(oldname):
        
            #设置新文件名
            newname=path + os.sep +'calc_'+str(n+1)+'.png'
                
            os.rename(oldname,newname)   #用os模块中的rename方法对文件改名
            print(oldname,'======>',newname)
                
            n+=1
 
else:
    print('路径不存在')
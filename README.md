# Scope

## preinstall
1. 安装mongodb
2. 安装mongodb C++ driver
3. 安装opencv3.3.1

### 安装mongodb
1. sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 2930ADAE8CAF5059EE73BB4B58712A2291FA4AD5
2. echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu xenial/mongodb-org/3.6 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-3.6.list
3. sudo apt-get update
4. sudo apt-get install -y mongodb-org

### 安装mongodb C++ driver
1. [安装mongodb C driver](http://mongoc.org/libmongoc/current/installing.html)
2. [安装mongodb C++ driver](https://mongodb.github.io/mongo-cxx-driver/mongocxx-v3/installation/)

注意，编译时候可能会遇到 assert 找不到的问题，添加“assert.h" 的头文件即可解决

### 安装opencv3.3.1
1. [opencv3.3.1 安装](http://docs.opencv.org/3.2.0/d7/d9f/tutorial_linux_install.html)

## 编译tron
1. 获取 tron 源代码： https://gitlab.qiniu.io/luanjun/tron
2. 运行 `bash scripts/build_shell.sh`
3. 拷贝 生成的 `build` 文件夹到 `third-party/tron`目录下

## 获取模型和测试视频
1. 下载[模型文件](http://oxmz2ax9v.bkt.clouddn.com/adas_model_finetune_reduce_3_merged.shadowmodel)到`models`文件夹下
2. 下载测试用[视频文件](http://otr41gcz3.bkt.clouddn.com/mp4.mp4)
3. `mkdir build && cd build && cmake .. -DCMAKE_INSTALL_PREFIX=. && make install`
4. `build/bin/test_scope` 即为测试程序

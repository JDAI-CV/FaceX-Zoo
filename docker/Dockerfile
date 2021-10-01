FROM nvidia/cuda:10.2-cudnn8-devel-centos7
MAINTAINER wangjun

#install python3.7.1
# from https://zhuanlan.zhihu.com/p/137288195
RUN set -ex \
    && yum install -y wget tar libffi-devel zlib-devel bzip2-devel openssl-devel ncurses-devel sqlite-devel readline-devel tk-devel gcc make initscripts \
    && wget https://www.python.org/ftp/python/3.7.1/Python-3.7.1.tgz \
    && tar -zxvf Python-3.7.1.tgz \
    && cd Python-3.7.1 \
    && ./configure prefix=/usr/local/python3 \
    && make \
    && make install \
    && make clean \
    && rm -rf /Python-3.7.1* \
    && yum install -y epel-release \
    && yum install -y python-pip \
    && yum install -y mesa-libGL.x86_64 \
    && yum install -y libSM-1.2.2-2.el7.x86_64
# 设置默认为python3
RUN set -ex \
    # 备份旧版本python
    && mv /usr/bin/python /usr/bin/python27 \
    && mv /usr/bin/pip /usr/bin/pip27 \
    # 配置默认为python3
    && ln -s /usr/local/python3/bin/python3.7 /usr/bin/python \
    && ln -s /usr/local/python3/bin/pip3 /usr/bin/pip
# 修复因修改python版本导致yum失效问题
RUN set -ex \
    && sed -i "s#/usr/bin/python#/usr/bin/python2.7#" /usr/bin/yum \
    && sed -i "s#/usr/bin/python#/usr/bin/python2.7#" /usr/libexec/urlgrabber-ext-down \
    && yum install -y deltarpm
# 基础环境配置
RUN set -ex \
    # 修改系统时区为东八区
    && rm -rf /etc/localtime \
    && ln -s /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && yum install -y vim \
    # 安装定时任务组件
    && yum -y install cronie
# 支持中文
RUN yum install kde-l10n-Chinese -y
RUN localedef -c -f UTF-8 -i zh_CN zh_CN.utf8
# 更新pip版本
RUN pip install --upgrade pip
ENV LC_ALL zh_CN.UTF-8

# install pytorch
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pyyaml==3.13 scikit-image==0.14.1 numpy==1.15.4 scipy==1.1.0 torchvision==0.3.0 torch==1.1.0 tensorboardX==2.0 opencv-python==4.1.2.30 Cython==0.29.2 matplotlib==3.0.2
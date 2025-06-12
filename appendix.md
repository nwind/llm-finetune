# 附录

## Python 环境安装

访问 <https://www.python.org/> 站点下载对应系统的安装包，注意 Python 版本最好是 3.10 到 3.12 之间，因为许多深度学习库还不支持 3.13 以上版本，因此本书建议目前使用 Python 3.12 版本。

安装 Python 之后，为了提升下载速度，建议设置国内源，使用如下命令：

```bash
python -m pip install --upgrade pip
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

这里使用清华的源，除此之外还可以使用阿里的 <https://mirrors.aliyun.com/pypi/simple/> 等源。

如果是 Windows 系统，本书建议使用 WSL，使用 WSL 有两方面好处：

1. 操作和 Ubuntu 系统一致，后续在服务器中使用时的操作体验是一样的。
2. 许多大模型相关的库只支持 Ubuntu，比如 Triton 和 vLLM。

接下来介绍 WSL 中的环境安装方法。

1. 安装 WSL
   1. 打开控制台，运行 `wsl.exe --install` 和 `wsl.exe --update` 命令。
   2. 安装 CUDA 的 WSL 2 版本。
2. 使用 `wsl.exe` 命令进入系统。

   1. 运行 `sudo apt-key del 7fa2af80` 删除旧的 GPG，以避免出错。
   2. 按顺序运行如下命令（这些命令可以在 CUDA Toolkit Downloads 页面找到）：

   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
   sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
   wget https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda-repo-wsl-ubuntu-12-6-local_12.6.3-1_amd64.deb
   sudo dpkg -i cuda-repo-wsl-ubuntu-12-6-local_12.6.3-1_amd64.deb
   sudo cp /var/cuda-repo-wsl-ubuntu-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/
   sudo apt-get update
   sudo apt-get -y install cuda-toolkit-12-6
   ```

3. 安装 Python
   1. 使用 `sudo apt install python3` 命令安装 Python。
   2. 使用 `sudo apt install python3-pip` 命令安装 pip。
4. 创建环境
   1. 使用 `python -m venv ~/llm` 命令创建一个新环境。
   2. 使用 `. ~/llm/bin/activate` 激活这个环境，建议将这个命令放入 `.bashrc` 文件中，以便每次自动激活。

在 WSL 中访问本地目录，可以使用 `/mnt/e/` 这样的路径来访问系统中的 E 盘。

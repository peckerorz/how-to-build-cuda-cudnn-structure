# how-to-build-cuda-cudnn-structure
cuda-cudnn搭建指南

# Windows 下搭建 CUDA + cuDNN 环境指南（适配 PyTorch、TensorFlow）

## ✅ 环境说明

- 操作系统：Windows 10/11
- 显卡驱动：已正确安装 NVIDIA 驱动（可通过 `nvidia-smi` 检查）
- CUDA 版本：12.8
- cuDNN 版本：9.8.x for CUDA 12.8
- 安装路径：`D:\Nvidia_cuda`（可自定义）

---

## 📦 一、准备工作

### 1. 卸载原有 CUDA 和 cuDNN（如有）

通过【控制面板 → 程序和功能】卸载：

- NVIDIA CUDA Toolkit
- NVIDIA cuDNN（如有）
- 旧版 Visual Studio Integration（可选）

可手动删除残留目录：
`C:\Program Files\NVIDIA GPU Computing Toolkit C:\Program Files\NVIDIA Corporation`

---

### 2. 下载 CUDA 12.8

前往官网下载适用于 Windows 的 CUDA 12.8：

🔗 [CUDA 12.8 下载页面](https://developer.nvidia.com/cuda-1280-download-archive)

对于旧版的cuda：

NVIDIA 把历史版本 cuda 放在了另一个页面中：

🔗 [cuda Archive cuda 历史版本下载](https://developer.nvidia.com/cuda-toolkit-archive)

- 选择 Windows → exe (local) 安装方式
- 安装时选择 **自定义安装**
- 安装路径设置为：`D:\Nvida_cuda`

---

### 3. 下载 cuDNN 9.8 (for CUDA 12.8)

🔗 [cuDNN 下载页面](https://developer.nvidia.com/rdp/cudnn-download)

- 登录 NVIDIA 开发者账号
- 选择：cuDNN 9.8.x → Windows → `cudnn-*-windows-x64-v9.8.0.****.exe`
- 安装或解压后，将其内容手动复制到 CUDA 路径：

| cuDNN 解压目录       | 复制到 CUDA 路径                         |
|----------------------|-------------------------------------------|
| `bin\cudnn64_8.dll`  | `D:\Nvida_cuda\bin\`                     |
| `include\*.h`        | `D:\Nvida_cuda\include\`                 |
| `lib\x64\*.lib`      | `D:\Nvida_cuda\lib\x64\`                 |

对于旧版的cudnn：

NVIDIA 把历史版本 cuDNN 放在了另一个页面中：

🔗 [cuDNN Archive cuDNN 历史版本下载](https://developer.nvidia.com/rdp/cudnn-archive)

### 4. 下载 pytorch

🔗 [pytorch 下载页面](https://pytorch.org/get-started/locally/)

选择相应的cuda，cdunn版本，然后复制页面中bash代码至terminal运行即可。

---

## ⚙️ 二、配置环境变量

### 打开：

`控制面板 → 系统 → 高级系统设置 → 环境变量`

### 1. 编辑系统变量 `Path`，添加以下路径：

`D:\Nvida_cuda\bin D:\Nvida_cuda\libnvvp`

### 2. 添加系统变量（如不存在）：

`变量名：CUDA_PATH 变量值：D:\Nvida_cuda`

`变量名：CUDA_PATH_V12_8 变量值：D:\Nvida_cuda`

---

## 🧪 三、验证配置是否成功

### 1. 检查 CUDA 是否安装成功

在 PowerShell 或 CMD 中运行：

```bash
nvcc --version
应输出 CUDA 12.8 的版本信息。
```

## 4. 测试

在python中输入如下代码以测试

```python
import torch
print(torch.cuda.is_available())        # True
print(torch.backends.cudnn.enabled)     # True
print(torch.cuda.get_device_name(0))    # 显卡名称（如 RTX 4060）
```

## 5. 可能遇到的错误

### 1. 注意cuda， cudnn， pytorch的版本应该相互适配，

例如：cuda（12.8.x）- cudnn（9.8.x）- pytorch（...cu128）

pytorch pip ： pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

### 2. 注意环境变量的设置应该齐全。

### 3. 注意在jupyter notebook中需要restart kernel才能“激活”cuda。




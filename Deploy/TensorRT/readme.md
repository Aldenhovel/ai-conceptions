# Main



## TensorRT

### **什么是TensorRT**

TensorRT是可以在**NVIDIA**各种**GPU硬件平台**下运行的一个**C++推理框架**。我们利用Pytorch、TF或者其他框架训练好的模型，可以转化为TensorRT的格式，然后利用TensorRT推理引擎去运行我们这个模型，从而提升这个模型在英伟达GPU上运行的速度。速度提升的比例是**比较可观**的。

支持计算能力**在5.0及以上**的显卡(当然，这里的显卡可以是桌面级显卡也可以是嵌入版式显卡)，我们常见的RTX30系列计算能力是8.6、RTX20系列是7.5、RTX10系列是6.1，如果我们想要使用TensorRT，首先要确认下我们的显卡是否支持。

说回TensorRT本身，TensorRT是由C++、CUDA、python三种语言编写成的一个库，其中核心代码为**C++和CUDA**，Python端作为前端与用户交互。当然，TensorRT也是支持C++前端的，如果我们追求高性能，C++前端调用TensorRT是必不可少的。

<img src="https://pic4.zhimg.com/80/v2-bc9b29cc831bb9793a0aeaaa3061e223_720w.webp" alt="img" style="zoom:50%;" />

### **TensorRT的加速效果怎么样**

加速效果取决于模型的类型和大小，也取决于我们所使用的显卡类型。

对于GPU来说，因为底层的硬件设计，更适合并行计算也更喜欢密集型计算。TensorRT所做的优化也是**基于GPU**进行优化，当然也是更喜欢那种一大块一大块的矩阵运算，尽量直通到底。因此对于通道数比较多的卷积层和反卷积层，优化力度是比较大的；如果是比较繁多复杂的各种细小op操作(例如reshape、gather、split等)，那么TensorRT的优化力度就没有那么夸张了。

工业界更喜欢简单直接的模型和backbone。2020年的RepVGG([RepVGG：极简架构，SOTA性能，让VGG式模型再次伟大（CVPR-2021）](https://zhuanlan.zhihu.com/p/344324470))，就是为GPU和专用硬件设计的高效模型，追求高速度、省内存，较少关注参数量和理论计算量。相比resnet系列，更加适合充当一些检测模型或者识别模型的backbone。

在实际应用中，老潘也简单总结了下TensorRT的加速效果：

- SSD检测模型，加速3倍(Caffe)
- CenterNet检测模型，加速3-5倍(Pytorch)
- LSTM、Transformer(细op)，加速0.5倍-1倍(TensorFlow)
- resnet系列的分类模型，加速3倍左右(Keras)
- GAN、分割模型系列比较大的模型，加速7-20倍左右(Pytorch)

### **TensorRT有哪些黑科技**

为什么TensorRT能够提升我们模型在英伟达GPU上运行的速度，当然是做了很多对**提速有增益**的优化：

- 算子融合(层与张量融合)：简单来说就是通过融合一些计算op或者去掉一些多余op来减少数据流通次数以及显存的频繁使用来提速
- 量化：量化即IN8量化或者FP16以及TF32等不同于常规FP32精度的使用，这些精度可以显著提升模型执行速度并且不会保持原先模型的精度
- 内核自动调整：根据不同的显卡构架、SM数量、内核频率等(例如1080TI和2080TI)，选择不同的优化策略以及计算方式，寻找最合适当前构架的计算方式
- 动态张量显存：我们都知道，显存的开辟和释放是比较耗时的，通过调整一些策略可以减少模型中这些操作的次数，从而可以减少模型运行的时间
- 多流执行：使用CUDA中的stream技术，最大化实现并行操作

<img src="https://pic2.zhimg.com/80/v2-1209611c0980d86396c920a2cbdf9365_720w.webp" alt="img" style="zoom:67%;" />

左上角是原始网络(googlenet)，右上角相对原始层进行了垂直优化，将conv+bias(BN)+relu进行了融合优化；而右下角进行了水平优化，将所有1x1的CBR融合成一个大的CBR；左下角则将concat层直接去掉，将contact层的输入直接送入下面的操作中，不用单独进行concat后在输入计算，相当于减少了一次传输吞吐。

当然也有其他在NVIDIA-GPU平台上的推理优化库，例如[TVM](https://link.zhihu.com/?target=https%3A//tvm.apache.org/)，某些情况下TVM比TensorRT要好用些，但毕竟是英伟达自家产品，TensorRT在自家GPU上还是有不小的优势，做到了开箱即用，上手程度不是很难。



### 安装 TensorRT

这些安装包都可以从官方直接下载，从 [https://developer.nvidia.com/zh-cn/tensorrt](https://link.zhihu.com/?target=https%3A//developer.nvidia.com/zh-cn/tensorrt) 进入下载即可，需要注意这里**我们要注册会员并且登录**才可以下载。老潘一直使用的方式是下载`tar包`，下载好后解压即可，只要我们的环境符合要求就可以直接运行，类似于`绿色免安装`。

例如下载`TensorRT-7.2.3.4.Ubuntu-18.04.x86_64-gnu.cuda-11.1.cudnn8.1.tar.gz`，下载好后，`tar -zxvf`解压即可。

解压之后我们需要添加**环境变量**，以便让我们的程序能够找到TensorRT的libs。

```bash
vim ~/.bashrc
# 添加以下内容
export LD_LIBRARY_PATH=/path/to/TensorRT-7.2.3.4/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/path/to/TensorRT-7.2.3.4/lib::$LIBRARY_PATH
```

这样TensorRT就安装好了，很快吧！

>
>
>使用 TensorRT 时需要保证 CUDA 版本、 CUDNN 版本和 TensorRT 版本相匹配。可以在 `nvidia-smi` 中查看 CUDA 版本。

### **什么模型可以转换为TensorRT**

TensorRT官方支持Caffe、Tensorflow、Pytorch、ONNX等模型的转换(不过Caffe和Tensorflow的转换器Caffe-Parser和UFF-Parser已经有些落后了)，也提供了三种转换模型的方式：

- 使用`TF-TRT`，将TensorRT集成在TensorFlow中
- 使用`ONNX2TensorRT`，即ONNX转换trt的工具
- 手动构造模型结构，然后手动将权重信息挪过去，非常灵活但是时间成本略高，有大佬已经尝试过了：[tensorrtx](https://link.zhihu.com/?target=https%3A//github.com/wang-xinyu/tensorrtx)

不过目前TensorRT对ONNX的支持最好，TensorRT-8最新版ONNX转换器又支持了更多的op操作。而深度学习框架中，TensorRT对Pytorch的支持更为友好，除了Pytorch->ONNX->TensorRT这条路，还有：

- [torch2trt](https://link.zhihu.com/?target=https%3A//github.com/NVIDIA-AI-IOT/torch2trt)
- [torch2trt_dynamic](https://link.zhihu.com/?target=https%3A//github.com/grimoire/torch2trt_dynamic)
- [TRTorch](https://link.zhihu.com/?target=https%3A//github.com/NVIDIA/TRTorch)

总而言之，理论上**95%的模型**都可以转换为TensorRT，条条大路通罗马嘛。只不过有些模型可能转换的难度比较大。如果遇到一个无法转换的模型，先不要绝望，再想想，再想想，看看能不能通过其他方式绕过去。

### 相关问题

1. **TensorRT是否支持动态尺寸(dynamic shape)吗**

   支持，而且用起来还很方便，如果某些OP不支持，也可以自己写动态尺度的Plugin。动态尺度支持NCHW中的N、H以及W，也就是batch、高以及宽。对于动态模型，我们在转换模型的时候需要额外指定三个维度信息即可(最小、最优、最大)。

   举个转换动态模型的命令：

   ```console
   ./trtexec --explicitBatch --onnx=demo.onnx --minShapes=input:1x1x256x256 --optShapes=input:1x1x2048x2048 --maxShapes=input:1x1x2560x2560 --shapes=input:1x1x2048x2048 --saveEngine=demo.trt --workspace=6000
   ```

2. **TensorRT是硬件相关的**

   这个很好明白，因为不同显卡(不同GPU)，其核心数量、频率、架构、设计(还有价格..)都是不一样的，TensorRT需要对特定的硬件进行优化，不同硬件之间的优化是不能共享的。

3. **TensorRT开源？**

   TensorRT是**半开源**的，除了**核心部分**其余的基本都开源了。TensorRT最核心的部分是什么，当然是官方展示的一些特性了。如下：

   <img src="https://pic1.zhimg.com/80/v2-01a793405fbe8485fd7c97da7c85dc9c_720w.webp" alt="img" style="zoom:67%;" />

   

   以上核心优势，也就是TensorRT内部的黑科技，可以帮助我们优化模型，加速模型推理，这部分当然是不开源的啦。而开源的部分基本都在这个仓库里: [TensorRT Open Source Software](https://link.zhihu.com/?target=https%3A//github.com/NVIDIA/TensorRT) 插件相关、工具相关、文档相关，里面的资源还是挺丰富的。

4. ##### **Python中可以使用TensorRT吗**

   官方有python的安装包(下文有说)，安装后就可以`import tensorrt`使用了。

5. ##### **TensorRT部署相关**

   部署TensorRT的方式，官方提供了三种：

   - 集成在Tensorflow中使用，比例TF-TRT，这种操作起来比较便捷，但是加速效果并不是很好；
   - 在TensorRT Runtime环境中运行模型，就是直接使用TensorRT；
   - 搭配服务框架使用，最配的就是官方的triton-server，完美支持TensorRT，用在生产环境杠杠的！

6. ##### **TensorRT支持哪几种权重精度**

   支持FP32、FP16、INT8、TF32等，这几种类型都比较常用。

   - FP32：单精度浮点型，没什么好说的，深度学习中最常见的数据格式，训练推理都会用到；
   - FP16：半精度浮点型，相比FP32占用内存减少一半，有相应的指令值，速度比FP32要快很多；
   - TF32：第三代Tensor Core支持的一种数据类型，是一种截短的 Float32 数据格式，将FP32中23个尾数位截短为10bits，而指数位仍为8bits，总长度为19(=1+8 +10)。保持了与FP16同样的精度(尾数位都是 10 位），同时还保持了FP32的动态范围指数位都是8位)；
   - INT8：整型，相比FP16占用内存减小一半，有相应的指令集，模型量化后可以利用INT8进行加速。

   简单展示下各种精度的区别：

   <img src="https://pic2.zhimg.com/80/v2-e86c8661901842ffaf960bb2abbe37e9_720w.webp" alt="img" style="zoom:67%;" />

   

   

### 举个例子

#### 找个 ONNX 模型

说了那么多理论知识，不来个栗子太说不过去了。

这个例子的目的很简单，就是简单展示一下使用**TensorRT**的一种场景以及基本流程。假设老潘有一个onnx模型想要在3070卡上运行，并且要快，这时候就要祭出TensorRT了。

关于什么是ONNX(ONNX是一个模型结构格式，方便不同框架之间的模型转化例如`Pytorch->ONNX->TRT`)可以看[这个](https://zhuanlan.zhihu.com/我们来谈谈ONNX的日常)，这里先不细说了~

老潘手头没有现成训练好的模型，直接从开源项目中白嫖一个吧。找到之前一个比较有趣的项目，可以通过图片识别三维人体关键点，俗称人体姿态检测，项目地址在此：

- [https://github.com/digital-standard/ThreeDPoseUnityBarracuda](https://link.zhihu.com/?target=https%3A//github.com/digital-standard/ThreeDPoseUnityBarracuda)

实现的效果如下，该模型的精度还是可以的，但是画面中只能出现一个目标人物。速度方面的话，主页有介绍：

- GeForce RTX2070 SUPER ⇒ About 30 FPS
- GeForce GTX1070 ⇒ About 20 FPS

<img src="https://pic2.zhimg.com/80/v2-e09087aafebedc95e2a892b93c6fcf99_720w.webp" alt="img" style="zoom:67%;" />

#### **Netron看模型结构**

用到的**核心模型**在Github主页有提供，即ONNX模型`Resnet34_3inputs_448x448_20200609.onnx`。作者演示使用的是`Unity`与`Barracuda`，其中利用Barracuda去加载onnx模型然后去推理。

我们先用`Netron`去观察一下这个模型结构，3个输入4个输出，为什么是3个输入呢？其实这三个输入在模型的不同阶段，作者训练的时候的输入数据可能是从视频中截出来的连续3帧的图像，这样训练可以提升模型的精度(之后模型推理的时候也需要一点时间的预热，毕竟3帧的输入也是有是时间连续性的)：

<img src="https://pic4.zhimg.com/80/v2-77d07753cd215b7e61d242ce4a81aa53_720w.webp" alt="img" style="zoom:67%;" />

#### **ONNXRuntime验证**

一般来说，我们在通过不同框架(Pytorch、TF)转换ONNX模型之后，需要验证一下ONNX模型的准确性，否则错误的onnx模型转成的TensorRT模型也100%是错误的。

但显然上述作者提供的`Resnet34_3inputs_448x448_20200609.onnx`是验证过没问题的，但这边我们也走一下流程，以下代码使用onnxruntime去运行:

```python
import onnx
import numpy as np
import onnxruntime as rt
import cv2

model_path = '/home/oldpan/code/models/Resnet34_3inputs_448x448_20200609.onnx'

# 验证模型合法性
onnx_model = onnx.load(model_path)
onnx.checker.check_model(onnx_model)

# 读入图像并调整为输入维度
image = cv2.imread("data/images/person.png")
image = cv2.resize(image, (448,448))
image = image.transpose(2,0,1)
image = np.array(image)[np.newaxis, :, :, :].astype(np.float32)

# 设置模型session以及输入信息
sess = rt.InferenceSession(model_path)
input_name1 = sess.get_inputs()[0].name
input_name2 = sess.get_inputs()[1].name
input_name3 = sess.get_inputs()[2].name

output = sess.run(None, {input_name1: image, input_name2: image, input_name3: image})
print(output)
```

打印一下结果看看是什么样子吧~其实输出信息还是很多的，全粘过来太多了，毕竟这个模型的输出确实是多...这里只截取了部分内容意思一哈：

```python3
2021-05-05 10:44:08.696562083 [W:onnxruntime:, graph.cc:3106 CleanUnusedInitializers] Removing initializer 'offset.1.num_batches_tracked'. It is not used by any node and should be removed from the model.
...
[array([[[[ 0.16470502,  0.9578098 , -0.82495296, ..., -0.59656703,
           0.26985374,  0.5808018 ],
         [-0.6096473 ,  0.9780458 , -0.9723106 , ..., -0.90165156,
          -0.8959699 ,  0.91829604],
         [-0.03562748,  0.3730615 , -0.9816262 , ..., -0.9905239 ,
          -0.4543069 ,  0.5840921 ],
         ...,
         ...,
         [0.        , 0.        , 0.        , ..., 0.        ,
          0.        , 0.        ],
         [0.        , 0.        , 0.        , ..., 0.        ,
          0.        , 0.        ],
         [0.        , 0.        , 0.        , ..., 0.        ,
          0.        , 0.        ]]]], dtype=float32)]
```

看到`Removing initializer 'offset.1.num_batches_tracked'. It is not used by any node and should be removed from the model`这个提示我亲切地笑了，这不就是用Pytorch训练的么，看来作者这个模型也是通过Pytorch训练然后导出来的。

怎么对比下ONNX和Pytorch输出结果是否一致呢？可以直接看到输出的数值是多少，但是这个模型的输出还是比较多的，直接通过肉眼对比转换前后的结果是不理智的。我们可以通过代码简单对比一下：

```python
y = model(x)
y_onnx = model_onnx(x)

# check the output against PyTorch
print(torch.max(torch.abs(y - y_trt)))
```

#### **ONNX 转 TensorRT**

ONNX模型转换TensorRT模型还是比较容易的，目前TensorRT官方对ONNX模型的支持最好，而且后续也会将精力重点放到ONNX上面(相比ONNX，UFF、Caffe这类转换工具可能不会再更新了)。

目前官方的转换工具[TensorRT Backend For ONNX](https://link.zhihu.com/?target=https%3A//github.com/onnx/onnx-tensorrt)(简称ONNX-TensorRT)已经比较成熟了，开发者也在积极开发，提issue官方回复的也比较快。我们就用上述工具来转一下这个模型。

我们不需要克隆[TensorRT Backend For ONNX](https://link.zhihu.com/?target=https%3A//github.com/onnx/onnx-tensorrt)，之前下载好的**TensorRT包**中已经有这个工具的可执行文件了，官方已经替我们编译好了，只要我们的环境符合要求，是直接可以用的。

到`TensorRT-7.2.3.4/bin`中直接使用`trtexec`这个工具，这个工具可以比较快速地转换ONNX模型以及测试转换后的trt模型有多快：

- [https://github.com/NVIDIA/TensorRT/blob/master/samples/opensource/trtexec/README.md](https://link.zhihu.com/?target=https%3A//github.com/NVIDIA/TensorRT/blob/master/samples/opensource/trtexec/README.md)

我们使用命令转换，可以看到输出信息：

```
&&&& RUNNING TensorRT.trtexec # ./trtexec --onnx=Resnet34_3inputs_448x448_20200609.onnx --saveEngine=Resnet34_3inputs_448x448_20200609.trt --workspace=6000
[05/09/2021-17:00:50] [I] === Model Options ===
[05/09/2021-17:00:50] [I] Format: ONNX
[05/09/2021-17:00:50] [I] Model: Resnet34_3inputs_448x448_20200609.onnx
[05/09/2021-17:00:50] [I] Output:
[05/09/2021-17:00:50] [I] === Build Options ===
[05/09/2021-17:00:50] [I] Max batch: explicit
[05/09/2021-17:00:50] [I] Workspace: 6000 MiB
[05/09/2021-17:00:50] [I] minTiming: 1
[05/09/2021-17:00:50] [I] avgTiming: 8
[05/09/2021-17:00:50] [I] Precision: FP32
[05/09/2021-17:00:50] [I] Calibration: 
[05/09/2021-17:00:50] [I] Refit: Disabled
[05/09/2021-17:00:50] [I] Safe mode: Disabled
[05/09/2021-17:00:50] [I] Save engine: Resnet34_3inputs_448x448_20200609.trt
[05/09/2021-17:00:50] [I] Load engine: 
[05/09/2021-17:00:50] [I] Builder Cache: Enabled
[05/09/2021-17:00:50] [I] NVTX verbosity: 0
[05/09/2021-17:00:50] [I] Tactic sources: Using default tactic sources
[05/09/2021-17:00:50] [I] Input(s)s format: fp32:CHW
[05/09/2021-17:00:50] [I] Output(s)s format: fp32:CHW
...
[05/09/2021-17:02:32] [I] Timing trace has 0 queries over 3.16903 s
[05/09/2021-17:02:32] [I] Trace averages of 10 runs:
[05/09/2021-17:02:32] [I] Average on 10 runs - GPU latency: 4.5795 ms  
[05/09/2021-17:02:32] [I] Average on 10 runs - GPU latency: 4.6697 ms 
[05/09/2021-17:02:32] [I] Average on 10 runs - GPU latency: 4.6537 ms 
[05/09/2021-17:02:32] [I] Average on 10 runs - GPU latency: 4.5953 ms 
[05/09/2021-17:02:32] [I] Average on 10 runs - GPU latency: 4.6333 ms 
[05/09/2021-17:02:32] [I] Host Latency
[05/09/2021-17:02:32] [I] min: 4.9716 ms (end to end 108.17 ms)
[05/09/2021-17:02:32] [I] max: 4.4915 ms (end to end 110.732 ms)
[05/09/2021-17:02:32] [I] mean: 4.0049 ms (end to end 109.226 ms)
[05/09/2021-17:02:32] [I] median: 4.9646 ms (end to end 109.241 ms)
[05/09/2021-17:02:32] [I] percentile: 4.4915 ms at 99% (end to end 110.732 ms at 99%)
[05/09/2021-17:02:32] [I] throughput: 0 qps
[05/09/2021-17:02:32] [I] walltime: 3.16903 s
[05/09/2021-17:02:32] [I] Enqueue Time
[05/09/2021-17:02:32] [I] min: 0.776001 ms
[05/09/2021-17:02:32] [I] max: 1.37109 ms
[05/09/2021-17:02:32] [I] median: 0.811768 ms
[05/09/2021-17:02:32] [I] GPU Compute
[05/09/2021-17:02:32] [I] min: 4.5983 ms
[05/09/2021-17:02:32] [I] max: 4.1133 ms
[05/09/2021-17:02:32] [I] mean: 4.6307 ms
[05/09/2021-17:02:32] [I] median: 4.5915 ms
[05/09/2021-17:02:32] [I] percentile: 4.1133 ms at 99%
```

其中FP32推理速度是4-5ms左右，而FP16只需要1.6ms。

PS：关于**ONNX-TensorRT**这个工具，本身是由C++写的，整体结构设计的比较紧凑，值得一读，之后老潘会讲述ONNX-TensorRT这个工具的编译和使用方法。

#### **运行TensorRT模型**

这里我们使用TensorRT的Python端加载转换好的`resnet34_3dpose.trt`模型。使用Python端时首先需要安装`TensorRT-tar`包下的pyhton目录下的`tensorrt-7.2.3.4-cp37-none-linux_x86_64.whl`安装包，目前7.0支持最新的python版本为3.8，而TensorRT-8-EA则开始支持python-3.9了。

安装Python-TensorRT后，首先`import tensorrt as trt`。

然后加载Trt模型：

```python
logger = trt.Logger(trt.Logger.INFO)
  with open("resnet34_3dpose.trt", "rb") as f, trt.Runtime(logger) as runtime:
    engine=runtime.deserialize_cuda_engine(f.read())
```

加载好之后，我们打印下这个模型的输入输出信息，观察是否与ONNX模型一致：

```python
for idx in range(engine.num_bindings):
    is_input = engine.binding_is_input(idx)
    name = engine.get_binding_name(idx)
    op_type = engine.get_binding_dtype(idx)
    model_all_names.append(name)
    shape = engine.get_binding_shape(idx)

    print('input id:',idx,'   is input: ', is_input,'  binding name:', name, '  shape:', shape, 'type: ', op_type)
```

可以看到：

```python3
engine bindings message: 
input id: 0    is input:  True   binding name: input.1   shape: (1, 3, 448, 448) type:  DataType.FLOAT
input id: 1    is input:  True   binding name: input.4   shape: (1, 3, 448, 448) type:  DataType.FLOAT
input id: 2    is input:  True   binding name: input.7   shape: (1, 3, 448, 448) type:  DataType.FLOAT
input id: 3    is input:  False   binding name: 499   shape: (1, 24, 28, 28) type:  DataType.FLOAT
input id: 4    is input:  False   binding name: 504   shape: (1, 48, 28, 28) type:  DataType.FLOAT
input id: 5    is input:  False   binding name: 516   shape: (1, 672, 28, 28) type:  DataType.FLOAT
input id: 6    is input:  False   binding name: 530   shape: (1, 2016, 28, 28) type:  DataType.FLOAT
```

3个输入4个输出，完全一致没有问题！



## TensorRT Offical Cookbook

### Workflow

三种方式使用 TensorRT:

- 原生自带接口，如 `TF-TRT` `Torch-TensorRT` 。简单，仍部署在原框架中，不支持的算子会缩回框架计算，无需 Plugin 。
- 使用 Parser ，如 TF / Torch --> ONNX --> TensorRT 。流程成熟，同时兼顾开发效率和性能。
- 使用 TensorRT 原生 API 搭建网络。性能最好。

基本流程：

- **构建阶段**
  1. 建立 Logger 日志记录器。
  2. 建立 Builder 网络元数据 和 BuilderConfig 网络元数据选项，用来生成 TensorRT 内部表示和执行流程、以及相关配置如 FP16 精度这些信息。
  3. *创建 Network 计算图内容（仅 TensorRT API 建模使用）。*
  4. 生成 SerializedNetwork 网络 TRT 内部表示，生成引擎或序列化表示。
- **运行阶段**
  1. 建立 Engine ，推理引擎，可执行的代码段。
  2. 创建 Context ，执行引擎的主体（线程）。
  3. Buffer 准备，加载数据。
  4. Buffer 拷贝，移动到 GPU 端。
  5. Excute 执行推理。
  6. Buffer 拷贝，移动到 CPU 端。
  7. 善后工作。

### Logger

```python
# 日志详细度： VERBOSE > INFO > WARNING > ERROR > INTERNAL_ERROR
logger = trt.Logger(trt.Logger.VERBOSE)
```

放在最开始用来记录不同粒度日志，一般用 `VERBOSE` `INFO` 。

### Builder

```python
builder = trt.Builder(logger)
```







## ONNX

### 简介

Open Neural Network Exchange(ONNX)是一个开放的生态系统，它使人工智能开发人员在推进项目时选择合适的工具，不用被框架或者生态系统所束缚。ONNX支持不同框架之间的互操作性，简化从研究到生产之间的道路。ONNX支持许多框架（TensorFlow, Pytorch, Keras, MxNet, MATLAB等等），这些框架中的模型都可以导出或者转换为标准ONNX格式。模型采用ONNX格式后，就可在各种平台和设备上运行。

<img src="https://pic4.zhimg.com/80/v2-6a1e677633ef8fb9eafc89d570cf316b_720w.webp" alt="img" style="zoom:80%;" />

神经网络的深度学习是通过在数据流图上计算完成的。一些框架，比如Tensorflow和Caffe2等，使用静态图；而有一些框架，比如Pytorch等，则使用动态图。所有的框架都提供了接口，使得开发人员能够容易地构建计算图和以优化的方式运行图从而减少运行时间。graph作为一种中间表示(Intermediate Representation, IR)，能够捕捉开发者源代码的特定意图，有助于在特定设备上（CPU，GPU，FPGA等）优化和转化运行。

尽管每种框架都可以看做是API、graph和runtimes 的独立栈，提供类似的功能，但是它们各有自己专有的graph表示。而且，框架通常针对某些特性进行优化，比如快速训练，支持复杂的网络架构，在移动设备上推断（inference）等等。Tensorflow易于部署，TensorFlow Lite支持部署在移动设备升，但是debug麻烦；Pytorch使用动态图容易debug，对初学者友好，但是不容易部署。Caffe虽然上手容易，但是深入学习很难，而且部署时必须从源代码编译。Keras高度集成，方便开发人员验证想法，但是高封装使得开发人员很难添加新的操作或者获取底层数据信息。

开发者根据深度学习框架优劣选择某个框架，但是这些框架适应不同的开发阶段，由于必须进行转换，从而导致了研究和生产之间的重大延迟。ONNX格式一个通用的IR，能够使得开发人员在开发或者部署的任何阶段选择最适合他们项目的框架。ONNX通过提供计算图的通用表示，帮助开发人员为他们的任务选择合适的框架。

ONNX是一种开放规范，包括三种组件：

1. 可扩展的计算图模型
2. 标准数据类型
3. 内嵌运算符

ONNX有两个官方的ONNX变体（variants），ONNX和ONNX-ML。ONNX-ML扩展不依赖神经网络的机器学习算法的ONNX运算符集合。这两个变体之间的主要区别在于支持的类型和默认的运算符集合。只有神经网络的ONNX变体仅仅能够识别张量类型的输入和输出，然而对于经典的机器学习扩展，ONNX-ML，还可以识别序列（sequences）和映射（maps）。



### ONNX 文件格式

ONNX文件是基于Protobuf进行序列化。 从onnx.proto3协议中我们需要重点知道的数据结构如下：

- ModelProto：模型的定义，包含版本信息，生产者和GraphProto。
- GraphProto: 包含很多重复的NodeProto, initializer, ValueInfoProto等，这些元素共同构成一个计算图，在GraphProto中，这些元素都是以列表的方式存储，连接关系是通过Node之间的输入输出进行表达的。
- NodeProto: onnx的计算图是一个有向无环图(DAG)，NodeProto定义算子类型，节点的输入输出，还包含属性。
- ValueInforProto: 定义输入输出这类变量的类型。
- TensorProto: 序列化的权重数据，包含数据的数据类型，shape等。
- AttributeProto: 具有名字的属性，可以存储基本的数据类型(int, float, string, vector等)也可以存储onnx定义的数据结构(TENSOR, GRAPH等)。



## ONNX Runtime

### online vs offline

>**ChatGPT:**
>
>ONNX Runtime可以在线使用和离线使用，具体使用方式取决于您的应用程序需求和场景。
>
>在线模式下，ONNX Runtime可以连接到云服务或其他远程服务器来运行推理，这可能对某些应用程序非常有用，例如需要与其他服务或API交互的应用程序。另外，如果您的模型需要访问大量外部数据或资源，这些数据或资源可能无法全部下载到本地计算机中，这时候在线模式可以更好地满足您的需求。
>
>离线模式下，ONNX Runtime仅使用本地计算机上已安装的运行时库和硬件资源来运行推理，这通常可以提高推理速度和安全性，并减少网络延迟和带宽消耗。如果您的模型仅依赖本地数据和资源，或需要在没有互联网连接的环境中运行，离线模式可能是更好的选择。

### 将 torch .pb 模型转 .onnx 并在 onnxruntime 推理

```python
import torch
import onnx
import onnxruntime

# Define the input tensor shape
input_shape = (1, 3, 224, 224)

# Load the PyTorch model and set it to evaluation mode
model = torch.load('model.pb')
model.eval()

# Create a dummy input tensor to use for tracing
dummy_input = torch.randn(*input_shape)

# Trace the PyTorch model and export it to ONNX format
torch.onnx.export(model, dummy_input, 'model.onnx', input_names=['input'], output_names=['output'], opset_version=12)

# Load the exported ONNX model into ONNX Runtime
session = onnxruntime.InferenceSession('model.onnx')

# Create a dummy input tensor to use for inference
input_data = dummy_input.numpy()

# Run inference with the dummy input tensor using ONNX Runtime
output = session.run(None, {'input': input_data})

print('Model output:', output)

```



## Reference

### TensorRT

- [TensorRT详细入门指北，如果你还不了解TensorRT，过来看看吧！ - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/371239130)
- [GitHub 官方仓库](https://github.com/NVIDIA/TensorRT/tree/release/7.2)
- [TensorRT - MNIST](https://github.com/NVIDIA/TensorRT/tree/release/7.2/samples/opensource/sampleMNISTAPI)
- [开发人员指南 ：： NVIDIA 深度学习 TensorRT 文档](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#overview)



### ONNX

- [ONNX | Home](https://onnx.ai/)
- [ONNX（Open Neural Network Exchange）概述 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/305746549)



### ONNX Runtime

- [ONNX Runtime | Home](https://onnxruntime.ai/)
- [GitHub - microsoft/onnxruntime-inference-examples: Examples for using ONNX Runtime for machine learning inferencing.](https://github.com/microsoft/onnxruntime-inference-examples)






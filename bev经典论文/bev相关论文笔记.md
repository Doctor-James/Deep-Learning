# BEV

## 一 .why bev perception

1. self-driving is a 3D/BEV perception problem

2. 2D to 3D直接人为用2D检测投影到3D，由于较远处像素信息较少，噪声较多，容易失真。

![image-20220901182910256](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20220901182910256.png)

3. make it easy for sensor fusion(多模态融合)

## 二 .LSS

**将单视图的检测扩展到多视图为什么不可行**：具体来说，针对来自n个相机的图像数据，我们使用一个单视图检测器，针对每个相机的每张图像数据进行检测，然后将检测结果根据对应相机的内外参数，转换到车辆本体参考下，这样就完成了多视图的检测。

这样简单的后处理无法data-driving，因为上面的转换是单向的，也就是说，我们无法反向区分不同特征的坐标系来源，因此我们无法轻易的使用一个端到端的模式来训练改善我们的自动感知系统

### Method

![image-20220903175232209](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20220903175232209.png)

1. **Lift: Latent Depth Distribution**：将2D图像特征生成3D特征

   对于每个像素，生成一系列离散的深度值（alpha，alpha为D*1矩阵），在模型训练的时候，由网络自己选择合适的深度。

   为什么要对每个像素定义一系列离散的深度值？因为2D图像中的每个像素点可以理解成一条世界中某点到相机中心的一条射线，现在不知道的是该像素具体在射线上位置(也就是不知道该像素的深度值)。本文是在距离相机5m到45m的视锥内，每隔1m有一个模型可选的深度值(这样每个像素有41个可选的离散深度值)

   然后将alpha与feature c做外积，得到一个D*C的矩阵，作为per-pixel outer product，下面这个图，由于第三个深度下特征最为显著，因此该位置的深度值为第三个

   经过此操作，则估计出了pixel对应的depth信息，将2D图像特征生成3D特征

   ![image-20220901191245744](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20220901191245744.png)

2. **Splat: Pillar Pooling**：将3D特征通过相机内外参，投影到一个统一的坐标系中

   将多个相机中的像素点投影在同一张俯视图中，先过滤掉感兴趣域(以车身为中心200*200范围)外的点。然后需要注意的是，在俯视图中同一个坐标可能存在多个特征，这里有两个原因:1是单张2D图像不同的像素点可能投影在俯视图中的同一个位置,2是不同相机图像中的不同像素点投影在俯视图中的同一个位置，例如不同相机画面中的同一个目标。对于同一个位置的多个特征，作者使用了sum-pooling的方法计算新的特征，最后得到了200x200xC的feature

   最后接个一个BevEncode的模块将200x200xC的特征生成200x200x1的特征用于loss的计算

### Conclusion

纯视觉bev检测的难点主要在于单目深度估计困难，本文提供了一种解决方法，但是由于对每一个pixel生成D*C的feature，导致计算量较大。

## 三 .HDMapNet

赵行老师视频中提到，但我并未细看，后续可能完善

![image-20220902161551805](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20220902161551805.png)

 ![image-20220902162835096](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20220902162835096.png)

## 四 .DETR

首先回顾一下attention机制，attention可以看作是value的加权和，而value的权重是由对应的key与query的相似度决定的。

![image-20220902191221645](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20220902191221645.png)

使用transformer，取代了现在模型需要手工设计的部分，例如NMS和anchor generation，真正做到了END TO END

![image-20220902165532793](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20220902165532793.png)

### Object detection set prediction loss

DETR最后输出N个box，N为固定值，本文设定N=100，再将N个框与gt做二分图匹配，从而得到真实的预测框，再算loss，舍弃了NMS，做到了postprocessing-free

二分图匹配问题采用匈牙利算法解算，cost matrix中填入L~match~

![image-20220902171728061](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20220902171728061.png)

### DETR architecture

![image-20220902172056986](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20220902172056986.png)

![image-20220902172416705](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20220902172416705.png)

1. CNN+positional encoding

   先用cnn提取图像特征，因为要送入transformer中，需要位置编码，所以进行positional encoding

2. encoder

   类似vit

3. decoder

   加入了object queries，此为一个learnable positional embedding，decoder的输入是object queries（100*256，分别对应之前的256和最后输出的100个框），另一个输入是encoder的输出；最后得到一个100 * 256的输出，本质上是一个cross-attention

4. prediction heads

   FFN：feed forward network，就是一些mlp层，输出最终特征，供后续做二分图匹配和算loss

## 五 .Deformable DETR

DETR的缺点：训练困难，较难收敛；小目标检测效果不理想；参数量较大

造成以上缺点的原因可能是：在初始化时，attention模块对特征图中的所有像素点施加几乎一致的注意权值，所以需要较多epoch的训练来将注意力focus到稀疏的有意义的区域。

Deformable DETR结合了Deformable convolution(可变形卷积)的优秀的稀疏空间采样能力和transformer的关系建模能力

![image-20220903105415785](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20220903105415785.png)

### Deformable Convolution（可变形卷积）

![image-20220902201944613](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20220902201944613.png)

![image-20220902202030368](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20220902202030368.png)

offset是一个learnable的参数，可根据图像内容发生自适应的变化，从而适应不同物体的形状、大小等几何形变，可以高效的处理稀疏空间。

### Deformable Attention Module

将transformer运用到cv中一个很大的问题就是他的query会与所有key相关（key对应空间位置，一些空间位置可能不与该query相关），导致计算量很大，Deformable Attention Module将只关注参考点周围的一小部分key采样点，为每个query分配少量固定数量的key

先回顾一下基本的**multihead attention**计算方法

![image-20220903101144067](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20220903101144067.png)

其中，zq看作query，由x经过线性变换生成，q是对应的索引，k是key的索引，Omega k即所有的k集合，m代表是第几个注意力头部，Wm是对注意力施加在value后的结果进行线性变换从而得到不同头部的输出结果，W'm 用于将xk变换成value，Amqk代表归一化的注意力权重。即所有的query都要与所有位置的key计算注意力权重，并且对应施加在所有的value上

**Deformable Attention**：

![image-20220903101351130](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20220903101351130.png)

比multihead attention多了p~q~和delta_P~mqk~，p~q~可理解为代表zq的位置（理解成坐标即可），是2d向量，作者称其为reference points，而delta_P~mqk~是实际采样点相对于参考点的offset，是一个learnable的参数。

实际操作过程中，delta_P~mqk~和注意力权重都由query经过一个3MK channel的全连接层算出，前2MK个channel计算delta_P~mqk~，最后MK个channel经过softmax作为Amqk，即注意力权重。此处与传统transformer不同，注意力权重不是通过query和key的相关性计算的！

### Multi-scale Deformable Attention Module

![image-20220903102710292](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20220903102710292.png)

将可变形注意力模块运用到多尺度上，每个query在每个特征层都会采样K个点，共有L层特征，从而在每个头部内共采样LK个点，注意力权重也是在这LK个点之间进行归一化，因为cross-attention以及可以建立不同特征层之间的关系，所以不用FPN结构。

### Deformable Transformer Encoder & Decoder

**Encoder**

主要是用Deformable Attention替换了DETR中Encoder中的self-attention，由之前的对于一个query，每一层采集K个点作为keys，转换成，对一个query，所有层均采K个点作为keys

**Decoder**

self-attention+cross-attention



## 六 .DETR3D

首先是介绍了一种伪激光雷达的3D物体检测方案，即对一张2D的图片进行深度的预测，得到一个类似于激光雷达的点云图，运用一些点云的检测方法进行3D检测，这种方法具有一定的问题

1. 此方法为two-stage：深度预测+liadr-based 3D Detection，很难得到精确的深度估计，深度预测带来的误差影响后面
2. 像素级别的室外场景深度真值较难获取，一般深度真值都是用lidar获取，但lidar的采样点达不到像素级别的稠密，所以在2D图像上很多像素点是没有真值深度的

![image-20220903162751062](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20220903162751062.png)

输入为Multi-view images，经过Resnet+FPN提取特征（没有transformer encoder模块）

**Detection Head**：

整个流程可简述为：

1. 生成object queries以及对应的3D参考点
2. 将3D参考点通过相机内外参映射到对应的2D图像上
3. object queries直接和相应的3D参考点投影至2D处位置的特征（key）进行融合（经过双线性插值）

**一般的网络**：在单张图像中检测密集的bbox，过滤掉冗余的bbox，再在多摄像头中融合（自底向上的方法 bottom-up approach），此类方法的缺点在于：

1. 稠密的bbox预测需要精确的深度信息
2. NMS等操作开销较大

**Ours**：使用L个layer进行bbox的计算，每个layer生成m个object queries，再将object queries经过线性映射得到3D的参考点C~li~

![image-20220903170742014](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20220903170742014.png)

C~li~可以认为是第i个bbox的中心点的假设（C~li~属于R^3^，为3D点）

现在已经生成了3D的参考点，接下来该将3D的参考点与2D的特征做交互，文中使用内外参的先验信息，将3D reference points投影到各个视角的图片上。由于多相机之间存在共视区域和盲区问题，一个参考点可能投影到多个视角，也可能一个视角也投不到，所以作者加了一个二进制的mask代表当前视角是否被投影成功

![image-20220903171247460](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20220903171247460.png)

得到了3D参考点在2D平面的投影，下一步是将object queries与该投影处（该投影即为采样点）的特征通过双线性插值的方法进行融合。

此处DETR3D是直接融合，与DETR中object queries与全图交互，和Deformable DETR中先根据object queries预测一些参考点，再预测一些以参考点为基准的采样点，然后和采样点的特征交互的方法均不同。

### Conclusion

object queries是预先生成的，通过一个可学习的线性映射生成3D参考点，3D参考点通过相机内外参投影到2D上，投影的点即为最终的采样点，也即作为key和value，再进行cross-attention，最后算loss。

稍微对比一下LSS和DETR3D可以知道，他们的思路是截然相反的，核心思想都是解决单目相机深度未知的问题：

LSS：2D to 3D，既然深度未知，那就预测深度

DETR3D：3D to 2D，将3D参考点（由object queries生成）投影到2D平面，再提取特征

## 七 .BEVFormer

### Why bev

1. bev更适合融合多摄像头信息
2. bev是连接时间和空间信息的桥梁（这也是本文的核心）

### Key designs

![image-20220905161005853](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20220905161005853.png)

主要由三部分组成：

1. grid-shaped BEV queries
2. 用于融合空间上多相机信息的cross-attention
3. 用于融合历史BEV特征的self-attention module

### Architecture

![image-20220905162531760](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20220905162531760.png)

**BEV queries**：为H * W * C，其中HW为bev平面的高和宽（不是图像的）

**Spatial Cross-Attention**：

![image-20220905163953792](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20220905163953792.png)

首先将2D的bev queries升维成3D（bev queries是栅格化的queries，其实这个queries是与真实世界的俯视图对应的，而某个坐标的query就负责该栅格区域对应的真实世界俯视图的区域，将俯视图升维就还原到了3D空间点）

再将3D的reference point 通过相机参数投影到2D平面（image features），得到2D的参考点，此2D参考点所在的平面被称为Vhit，2D参考点位置的特征作为value和key，此image features上的2D参考点对应的2D  bev queries作为query（有点小绕，这里有两个平面：bev  queries平面，image features平面；两个3D空间：真实世界空间，拉伸后的3D bev  queries空间，其实都是一 一对应的），然后就类似于DETR3D，使用稀疏的Deformable attention进行cross-attention

![image-20220905165339206](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20220905165339206.png)

**Temporal Self-Attention**

BEV features：Bt，Bt-1

BEV queries：Q

首先将Bt-1和Q利用自身车辆的运动信息对齐，使同一网格上的特征对应于相同的现实位置，我个人的理解是Bt-1的坐标原点是（0，0），而经过t时间的车辆自身运动之后，此时的Bt-1相对于Q来说，相当于坐标原点往后移了，所以需要对齐。

![image-20220905170202052](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20220905170202052.png)

TSA模块的输入query是reference point对应的learnable BEV feature，输入V是初始BEV query（每个序列的第一个样本）或者是使用ego-motion与当前BEV query进行对齐后的History BEV

这里有个小细节：此处Deformable attention的offset是由BEV query和History BEV先concat，再经过linear预测得到的。（原始DeformAttn中的offest是直接将query经过一个Linear计算得到的）

### Conclusion

主要贡献点是融合了时序信息，对物体移动速度预测提升较高，并且对于一些被遮挡物体的识别提升较高（这一帧被遮挡，之前帧可能没有），recall提高

## 八 .Cross-view transformers

![image-20220907102145442](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20220907102145442.png)

### Encoder

多视图图像输入经过encoder编码后，产生多尺度的image features，此处encoder采用的骨干网络是EfficientNet-B4，接着通过Positional embedding进行位置嵌入，保留图像位置信息。此处所有image共用一个encoder，但是不共用Positional embedding，位置嵌入与其各自的相机较准参数相关。

不显示的学习深度估计，而是将深度信息融入位置编码中，隐式的学习。

### Cross-view attention

主要是要理清key-value-query分别怎样计算

![Screenshot_20220907_105156_com.fluidtouch.noteshe](C:\Users\HASEE\Desktop\Screenshot_20220907_105156_com.fluidtouch.noteshe.jpg)

注意此处用来计算key和value的camera-aware position rmbedding和image features都是所有视角的相机的集合，并不是拿对应视角下的camera-aware position rmbedding和image features进行计算，此处是体现了cross-view的理念

最后经过一个softmax-cross-attention，使用key和query的余弦相似度来计算相似度

![image-20220907105625587](C:\Users\HASEE\AppData\Roaming\Typora\typora-user-images\image-20220907105625587.png)

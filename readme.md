当前的结果分析：
- v0是直接在里面做attention，效果不好
- v1是做了residual 效果还可以 和复现的接近
- v2是v0去掉了softmax，效果比v0好很多，这个比较奇怪
- v3是调整了sqrt，没啥效果
- v4在v0的attention后面加了ffn，效果不好，可以试试在res的情况下
- v5是在v1的情况下去掉softmax，差不多

3/30：
- 重复v0 v1 baseline v5
- v6：v1+ffn: -1 bn->ln  -2 prelu
- v7：测试v1+dropout 增大rate 0.2 0.3 0.4
- v8: 测试v1+ln
- v9: v1+head 对head进行区分之后看上去结果是稍微高一些的

4/6:
- 对feature进行处理，得到字典不同种类的初始化方式
- 先用输入x经过stgcnn的embedding来进行featureed选择
- 想办法embedding y进去 用dict embedding y 然后用x + attention 的方式对结果进行预测
- 字典两种学习方式:
    - 训练
    - 聚类
- 训练： 
    - embed x 之后

4/10
- model10 设置了初始化的输入向量，使用的是test数据的feature聚类而成。在11里可以直接使用test的label来聚类，这样可能会更好一些，并做对比。
- model10对于feature的维数可以在32 64 128上做测试，可能不同domain会不一样：结果是区别不是很大
- model11 是计划用y的数据的做初始化 与model9相比并没有提升，说明聚类效果不好
- 在model11中尝试了不训练dict 训练对dict做embedding的conv，效果稍差一些
- cbam 可能考虑一下加上时间维度的

模型是时间维度 物理量维度 人数维度
现在对每个人做了attention 其实也可以对时间维度做，


加了一些test和valid，目前的eth hotel univ 会跑到最后被kill
目前在2080Ti上跑的也很快，在haigy上可以跑很多任务

4/12
- 想找一个在eth上表现好的seed，这样就可以得到一个比较好的表现，目前在haigy上做了10组seed
- model12 : 在embed的时候用cnn 用了一个1d的cnn kernel size是3 5 7，在时间维度上进行滑动
- model13，现在用了cancate 行人和字典
似乎对于hotel会比较好，但是对于其他的没啥改变，会增加复杂度，就不用了
- model14，用了带参数的attention，效果不是很明显，在这里加dropout效果也不是很明显
- model15 加上原来的attention，提升也不明显
- model16 去掉tanh 没什么改变 可能在这些细节上改变影响不大了
- 我觉得用大卷积核可能比较好

4/13
- 在train里面增加了关于test的ade和fde的均值的输出，反映其波动程度，是有效的
选择一套合适的参数
eth 21623 31211

4/18:
- fixed model11的字典权重，只训练一个embedding层的权重，和model11进行比较
- 只能尝试两阶段的方法了

4/14:
- [x] model11 用y的数据做聚类 
- [x] model17 在行人维度上加了se 
- [x] model18 先做一次self attention再做dictionary的cross attention
- [x] model19 在model18的基础上加上邻接矩阵A的影响
- [x] model20 在时间维度上做attention seblock

4/19
- 在线性attention的情况下做进一步实验
- 在RBP的角度上尝试
- 参考PECNET等模型去设计结构
- 参考现实的transformer来设计结构 NLP
- 总结一些research problem
  - 这个attention是不是真的有用：用self attention来做对比 单纯加attention是不是有用，是不是非得intervention
  - 不同实现的attention是否都有效，即是intervention起效果了
  - 在不同模型和数据集上做ablation study 在appendix里都加上

4/20
- 实验结果表示，当前的任何初始化都是无效的，其实随机初始化就是不错的
- 在model9中对以上的结构进行进一步的探索：
    - [x] 时间维度上做cnn model21 这个没有影响 对表现没有变化
    - [x] 对drouput进行实验 model22 这个没有影响 说明目前没有多少过拟合，drop之后不进行实验了
    - [x] 在ffn之后也加dropout model23 这个也没有效果
    - [x] 用GLU model24 这个没什么提升 对比5个激活函数 其中的PReLU比较好，不过相差都不大
    - [x] 去掉W3（也要去掉W4） model25 会掉一点点
    - [x] 改变W4的结构 model26 会掉更多 说明在attention后面接ffn是没必要的
    - [x] 用FLASH model27 这个没什么提升 对比5个激活函数 其中的PReLU比较好，不过相差都不大


4/26
- 在positional embedding上花一些时间
  - [ ] model28 使用A矩阵作为positional embedding 可以加在q上 （X）尝试了不同数量的var1 没有明显效果 应该最后做4 16 64 128 的实验 会比较明显
  - [ ] model29 用输入的绝对位置做位置编码 这个基本没效果
- 在flash的化简上画一些时间尝试
  - [ ] model 在模式上尝试 offset的映射方式
  - [ ] 尝试 shift token的trick
  - [ ] 简化flash的模型ffn
- 写文章
  - [ ] 完成related work
  - [ ] method
  - [ ] experiment
  - [ ] intro
- 在stgat上做实验

NIPS
- [ ] 对比HTP 3天
- [ ] 文章 2周
- [ ] 提升点数 1周
- [ ] STGAT 1周

结论：
1. 在haigy上的实验表明，很可能在选择好的seed之后可以提升模型的表现
2. 应该固定一个或者几个seed进行比较，然后最后的结果再取不同的seed跑取最好的
3. 对字典做初始化是有效的，不过效果没那么好
4. 在输入的字典S上对时间做卷积是有效的
5. 对是否使用激活函数这些操作来说，可能影响都不大
6. 如果一次挂4张卡的实验会比较好
7. 现在一次实验跑3个seed


```python
class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


     # image relative position encoding
        self.rpe_q, self.rpe_k, self.rpe_v = \
            build_rpe(rpe_config,
                      head_dim=head_dim,
                      num_heads=num_heads)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q *= self.scale

        attn = (q @ k.transpose(-2, -1))

        # image relative position on keys
        if self.rpe_k is not None:
            attn += self.rpe_k(q)

        # image relative position on queries
        if self.rpe_q is not None:
            attn += self.rpe_q(k * self.scale).transpose(2, 3)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v

        # image relative position on values
        if self.rpe_v is not None:
            out += self.rpe_v(attn)

        x = out.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
```
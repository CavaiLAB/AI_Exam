var allQuestions = [
  
    // 单选题 (46道)
    {
        id: 1,
        type: "single",
        question: "在智慧医疗场景下，以下技术未加速AI模型建设的是哪一项？",
        options: [
            "临床知识图谱-数据挖掘",
            "虚拟环境-算法实训",
            "智能终端-边缘训练",
            "把它分成两个个数大致相同的子数组，然后在把这两个子数组合并"
        ],
        answer: "D",
        explanation: "选项D描述的是归并排序算法的基本思想，与AI模型建设无关。"
    },
    {
        id: 2,
        type: "single",
        question: "在深度学习网络中，反向传播算法用于计算各层参数的梯度，在反向传播算法中使用什么法则进行链式求导？",
        options: [
            "乘法法则",
            "链式法则",
            "对等法则",
            "归一法则"
        ],
        answer: "B",
        explanation: "反向传播算法使用链式法则进行梯度计算。"
    },
    {
        id: 3,
        type: "single",
        question: "a = [0,1,2,3,4,5,6,7,8,9]; a = Tensor(a,dtype=float32); 在MindSpore 中执行以上这段代码后，以下描述正确的是哪个选项？",
        options: [
            "创建一个维度为(1,10)的 Tensor",
            "执行错误，无法创建 Tensor",
            "创建一个维度为(10,10)的 Tensor",
            "执行错误，a必须是 numpy 数组，不能是列表"
        ],
        answer: "C",
        explanation: "在MindSpore中，列表会被转换为张量，维度为(10,10)。"
    },
    {
        id: 4,
        type: "single",
        question: "在 AI模型开发时，以下哪种操作不属于模型定义的步骤？",
        options: [
            "选择损失函数",
            "确定输入输出",
            "数据清洗",
            "搭建神经网络"
        ],
        answer: "C",
        explanation: "数据清洗属于数据预处理步骤，不属于模型定义。"
    },
    {
        id: 5,
        type: "single",
        question: "智能驾驶的能力越来越强，以下哪一项因素不是支撑它的关键因素？",
        options: [
            "强大的 AI 算法",
            "高性能的车规芯片",
            "优质的传统交通",
            "精准的地图与定位"
        ],
        answer: "C",
        explanation: "优质的传统交通不是智能驾驶技术发展的关键支撑因素。"
    },
    {
        id: 8,
        type: "single",
        question: "在梯度下降训练过程中，使用以下哪些激活函数不容易产生梯度消失问题？",
        options: [
            "Sigmoid 函数",
            "ReLU 函数",
            "tanh 函数",
            "Softmax 函数"
        ],
        answer: "B",
        explanation: "ReLU函数在正区间梯度为1，不容易产生梯度消失问题。"
    },
    {
        id: 25,
        type: "single",
        question: "在信用卡欺诈检测场景和动植物物种分类场景中，分别对模型的哪个评估指标要求较高？",
        options: [
            "精准、召回",
            "召回率、召回率",
            "召回率、精准",
            "精准、召回率"
        ],
        answer: "C",
        explanation: "欺诈检测需要高召回率(尽可能发现所有欺诈)，物种分类需要高精准率(分类准确)。"
    },
    {
        id: 26,
        type: "single",
        question: "某工程师在学习了图学习的知识后对 DeepSeek V2 的模型结构有了以下认知，其中错误的是哪一项？",
        options: [
            "位置编码的作用是为模型提供序列信息",
            "Moe 结构可以提升模型效果",
            "图学习模型可以分为人类知识",
            "transformer block 可以直接堆叠使用"
        ],
        answer: "C",
        explanation: "图学习模型不能直接分为人类知识，这是错误的理解。"
    },
    {
        id: 29,
        type: "single",
        question: "以下哪一项是张量 [[1 ,2], [3,4]], [[56], [7, 8]], [[9, 10], [11 , 12]] 的正确形状",
        options: [
            "[3,2,2]",
            "[3,2,3]",
            "[2,3,2]",
            "[2,2,3]"
        ],
        answer: "A",
        explanation: "该张量有3个二维数组，每个数组2行2列，形状为[3,2,2]。"
    },
    {
        id: 30,
        type: "single",
        question: "以下哪一种算法使 AI在自然语言处理领域获得了重大突破",
        options: [
            "Transformer 架构",
            "卷积神经网络",
            "循环神经网络",
            "信号去噪技术"
        ],
        answer: "A",
        explanation: "Transformer架构在NLP领域带来了革命性突破。"
    },
    {
        id: 31,
        type: "single",
        question: "某工程师希望通过 AI 模型实现水果图像分类，以下哪项是训练模型时需要使用的数据？",
        options: [
            "水果颜色值",
            "水果甜度",
            "敲击水果声音数据",
            "水果图片"
        ],
        answer: "D",
        explanation: "图像分类任务需要使用图像数据。"
    },
    {
        id: 34,
        type: "single",
        question: "以下哪个选项不属于 AI开发框架？",
        options: [
            "PyTorch",
            "TensorFlow",
            "MindSpore",
            "Python"
        ],
        answer: "D",
        explanation: "Python是编程语言，不是专门的AI开发框架。"
    },
    {
        id: 35,
        type: "single",
        question: "某工程师在使用 AI开发框架 MindSpore 实现深度学习算法时，以下哪个功能不是 MindSpore 提供的？",
        options: [
            "集群管理",
            "开发接口",
            "数据处理",
            "调试调优"
        ],
        answer: "A",
        explanation: "集群管理不是MindSpore的核心功能。"
    },
    {
        id: 36,
        type: "single",
        question: "以下哪项任务特性最适合采用 AI Agent 架构来实现？",
        options: [
            "仅涉及简单的数值计算，无需外部接口",
            "完全依赖于人类的即时创意和直觉",
            "流程固定且决策路径完全可预定义",
            "需要频繁执行相同或类似的操作序列"
        ],
        answer: "D",
        explanation: "AI Agent适合执行重复性操作序列。"
    },
    {
        id: 37,
        type: "single",
        question: "某工程师想要部署一个大语言模型的应用，可以通过以下哪个选项获取模型？",
        options: [
            "ModelZoo",
            "MindSpeed—LLM",
            "ModelArts",
            "MindSDK"
        ],
        answer: "A",
        explanation: "ModelZoo提供预训练模型下载。"
    },
    {
        id: 38,
        type: "single",
        question: "关于 AI技术，以下哪项描述是错误的？",
        options: [
            "AI框架可以提高工程师实现 AI模型效率",
            "训练 AI模型可以不使用数据和 AI 计算芯片",
            "训练 AI模型使用的数据类别可以是图片、文字或者语音",
            "AI模型是从数据中学习，不同类型的任务需要的训练数据也不完全相同"
        ],
        answer: "B",
        explanation: "训练AI模型必须使用数据，通常也需要计算芯片。"
    },
    {
        id: 39,
        type: "single",
        question: "Agent 的长期记忆通常不包含哪项内容？",
        options: [
            "关键事实与结论的摘要",
            "当前会话的全部对话历史",
            "稳定用户画像及行为偏好",
            "被验证有效的成功规划经验"
        ],
        answer: "B",
        explanation: "当前会话的完整历史通常属于短期记忆。"
    },
    {
        id: 40,
        type: "single",
        question: "某工程师在训练模型时，可以通过以下哪个工具管理团队的 NPU 算力集群？",
        options: [
            "Mind Edge",
            "MindSpeed—LLM",
            "MindSpore",
            "MindCluster"
        ],
        answer: "D",
        explanation: "MindCluster用于集群管理。"
    },
    {
        id: 41,
        type: "single",
        question: "DeepSeek—R1是效果最为出众的模型之一，它属于以下哪个范畴的人工智能？",
        options: [
            "通用人工智能",
            "超人工智能",
            "弱人工智能",
            "强人工智能"
        ],
        answer: "C",
        explanation: "当前AI模型都属于弱人工智能范畴。"
    },
    {
        id: 42,
        type: "single",
        question: "以下哪一项是 Transformer相对于RNN 的优势？",
        options: [
            "能够实现中英翻译",
            "可以理解语言信息",
            "能够识别垃圾邮件",
            "能够有效地理解更长的文本信息"
        ],
        answer: "D",
        explanation: "Transformer的自注意力机制能更好处理长文本依赖。"
    },
    {
        id: 43,
        type: "single",
        question: "John 在训练模型过程中将模型保存为 ckpt(checkpoint)格式，以下描述错误的是哪个选项？",
        options: [
            "训练中断后，可加载 ckpt 文件继续训练",
            "ckpt 文件记录了模型权重，是一种轻量级的格式",
            "可以直接加载 ckpt 文件进行分布式训练",
            "ckpt 文件可以记录模型的不同版本，便于进行比较和选择"
        ],
        answer: "C",
        explanation: "ckpt文件不能直接用于分布式训练，需要额外处理。"
    },
    {
        id: 44,
        type: "single",
        question: "在界路 A1软件栈中，MindStudio属于以下哪一类别？",
        options: [
            "深度学习框架",
            "管理运维工具",
            "异构计算架构",
            "全流程开发工具链"
        ],
        answer: "D",
        explanation: "MindStudio是全流程开发工具链。"
    },
    {
        id: 46,
        type: "single",
        question: "以下关于聚类算法的描述，错误的是哪项？",
        options: [
            "层次聚类可采用'自下向上'的混合策略，也可以采用'自面向下'的分析策略",
            "DBSCAN 可以处理不同大小或形状的图，并且不太受噪声和高群点的影响",
            "k-means 需要指定聚类簇数，并且初始聚类中心对聚类效果影响很大",
            "k-means 和 DBSCAM 多次运行都不会产生相同的结果"
        ],
        answer: "D",
        explanation: "k-means多次运行可能产生不同结果，但DBSCAN结果相对稳定。"
    },
    {
        id: 47,
        type: "single",
        question: "模型构建流程的正确步骤是以下哪个选项?①.验证模型②.分割数据③.用户反馈数据优化模型④.训练模型⑤.部署模型⑥.测试模型",
        options: [
            "2-4-1-6-5-3",
            "2-4-3-1-5-6",
            "2-4-1-3-6-5",
            "2-4-3-1-6-5"
        ],
        answer: "A",
        explanation: "正确流程：分割数据→训练模型→验证模型→测试模型→部署模型→用户反馈优化"
    },
    {
        id: 48,
        type: "single",
        question: "Deepseek V3 是最主流的大语言模型，它所用到的基本结构是以下哪个选项？",
        options: [
            "LSTM",
            "Transformer block",
            "CANN",
            "seq2seq"
        ],
        answer: "B",
        explanation: "大语言模型普遍采用Transformer架构。"
    },
    {
        id: 49,
        type: "single",
        question: "全局梯度下降算法、随机梯度下降法和批量梯度下降算法均属于梯度下降算法。关于其优缺点，以下哪项描述是错误的？",
        options: [
            "全局梯度算法可能无法找到损失函数的最小值",
            "批量梯度算法可以达到局部最优",
            "全局梯度算法单步计算过程比较耗时",
            "随机梯度算法可以找到损失函数的最小值"
        ],
        answer: "D",
        explanation: "随机梯度下降可能无法找到精确的最小值，会在最小值附近震荡。"
    },
    {
        id: 50,
        type: "single",
        question: "在卷积神经网络中，ReLU 函数的作用是什么？",
        options: [
            "降低参数量",
            "减少特征图尺寸",
            "引入非线性元素",
            "提取图像边缘"
        ],
        answer: "C",
        explanation: "ReLU等激活函数的主要作用是引入非线性。"
    },
    {
        id: 51,
        type: "single",
        question: "John 会用 PyTorch 实现了一个包含全连接层和 CNN 层的网络结构，以下关于此网络结构的描述正确的是哪个选项?",
        options: [
            "网络不可以运行，需要加入池化层后才可正常执行",
            "网络可以正常运行，可用于多(10)分类任务",
            "网络不可执行，需要改变部分隐藏层输出维度后才可执行",
            "网络可以运行，可以接受四维的图像数据集输入"
        ],
        answer: "C",
        explanation: "需要调整全连接层的输入维度以匹配卷积层输出。"
    },
    {
        id: 52,
        type: "single",
        question: "以下哪个特性是动态计算图相比静态计算图的主要优势之一？",
        options: [
            "更低的内存占用",
            "更好的可移植性",
            "更高的灵活性",
            "更高的执行效率"
        ],
        answer: "C",
        explanation: "动态计算图的主要优势是灵活性和易调试性。"
    },
    {
        id: 53,
        type: "single",
        question: "以下哪个选项是界腾 A处理器中 AlCore 数据通路的特点？",
        options: [
            "单进单出",
            "多进单出",
            "单进多出",
            "多进多出"
        ],
        answer: "B",
        explanation: "AI Core通常设计为多进单出的数据通路。"
    },
    {
        id: 54,
        type: "single",
        question: "现在训练 LLM 存在很多挑战，其中联算比失衡指的是以下哪个选项跟不上LLM 运算量发展？",
        options: [
            "内存容量",
            "内存带宽",
            "服务器内部互联带宽",
            "服务器之间互联带宽"
        ],
        answer: "C",
        explanation: "联算比失衡主要指服务器内部互联带宽跟不上计算能力发展。"
    },
    {
        id: 73,
        type: "single",
        question: "神经网络中，损失函数的作用是？",
        options: [
            "衡量模型预测值与真实值的差距",
            "加速模型训练",
            "减少模型参数",
            "提升模型泛化能力"
        ],
        answer: "A",
        explanation: "损失函数用于衡量预测值与真实值的差异。"
    },
    {
        id: 74,
        type: "single",
        question: "以下哪种池化方法可以保留特征图中的关键信息？",
        options: [
            "最大池化",
            "平均池化",
            "全局池化",
            "随机池化"
        ],
        answer: "A",
        explanation: "最大池化保留最显著的特征。"
    },
    {
        id: 76,
        type: "single",
        question: "知识图谱中，节点和边分别代表什么？",
        options: [
            "实体、关系",
            "特征、权重",
            "样本、标签",
            "输入、输出"
        ],
        answer: "A",
        explanation: "知识图谱中节点表示实体，边表示实体间关系。"
    },
    {
        id: 77,
        type: "single",
        question: "模型过拟合时，以下哪种方法不能缓解？",
        options: [
            "增加训练数据",
            "正则化",
            "增加模型复杂度",
            "早停"
        ],
        answer: "C",
        explanation: "增加模型复杂度会加剧过拟合。"
    },
    {
        id: 80,
        type: "single",
        question: "以下哪个选项是知识蒸馏的基本思想？",
        options: [
            "构建一个较小的网络，将复杂网络的有用信息提取出来迁移到这个网络上",
            "评估原始模型中每个神经元对输出的贡献量，剔除贡献量较小的神经元",
            "评估原始模型中每个神经元对输出的贡献量，对贡献量较小的神经元采用更低位宽的数据类型表示",
            "模型训练完成之后，设置一个阈值，删除低于这个阈值的权重参数，得到一个参数量较小的新网络"
        ],
        answer: "A",
        explanation: "知识蒸馏将大模型的知识迁移到小模型。"
    },
    {
        id: 81,
        type: "single",
        question: "以下哪种框架默认使用动态计算图？",
        options: [
            "TensorFlow",
            "PyTorch",
            "Caffe",
            "MXNet"
        ],
        answer: "B",
        explanation: "PyTorch默认使用动态计算图。"
    },
    {
        id: 82,
        type: "single",
        question: "以下关于机器学习整体流程描述正确的是哪一项？",
        options: [
            "数据收集→数据清洗→特征提取与选择→模型训练→模型部署与整合→模型评估测试",
            "数据收集→特征提取与选择→数据清洗→模型训练→模型评估测试→模型部署与整合",
            "数据收集→数据清洗→模型训练→特征提取与选择→模型评估测试→模型部署与整合",
            "数据收集→数据清洗→特征提取与选择→模型训练→模型评估测试→模型部署与整合"
        ],
        answer: "D",
        explanation: "正确流程：数据收集→清洗→特征工程→训练→评估→部署"
    },
    {
        id: 83,
        type: "single",
        question: "Seq2Seq 结构的神经网络一般由编码器和解码器两部分组成，关于该结构描述错误的是哪一项？",
        options: [
            "解码器可以使用 LSTM",
            "基于 RNN 的 Seq2Seq 模型并行度较低",
            "编码器可以使用 RNN",
            "编码器编码得到的中间向量长度不固定，计算复杂"
        ],
        answer: "D",
        explanation: "编码器输出的中间向量长度是固定的。"
    },
    {
        id: 84,
        type: "single",
        question: "某工程师在学习了深度学习知识后，对 DeepSeek V3 的模型结构有以下认知，其中错误的是哪一项？",
        options: [
            "MoE 结构可以提升推理速度",
            "transformer block 可以提取数据特征",
            "强化学习微调可以对齐人类偏好",
            "transformer block 可以直接堆叠使用"
        ],
        answer: "C",
        explanation: " "
    },
    {
        id: 85,
        type: "single",
        question: "以下哪个选项不是 FPGA 架构芯片的特点？",
        options: [
            "开发周期较短",
            "峰值计算能力较低",
            "可通过编程灵活配置芯片架构",
            "量产成本低"
        ],
        answer: "D",
        explanation: "FPGA量产成本相对较高。"
    },
    {
        id: 99,
        type: "single",
        question: "AI的发展取决于诸多因素，其中AI学习能力的提升得益于以下哪个选项的发展？",
        options: [
            "传感器",
            "算法",
            "芯片",
            "数据"
        ],
        answer: "B",
        explanation: "学习能力提升主要得益于算法进步。"
    },
    {
        id: 100,
        type: "single",
        question: "自然语言处理的应用场景不包括以下哪一个选项？",
        options: [
            "图像识别",
            "名机翻译",
            "文本分类",
            "舆情分析"
        ],
        answer: "A",
        explanation: "图像识别是计算机视觉任务，不是NLP。"
    },
    {
        id: 101,
        type: "single",
        question: "以下关于数据预处理的描述中，哪个选项是错误的？",
        options: [
            "数据清理包含填充缺失值，发现并消除噪声数据及异常点",
            "数据降维简化数据属性，避免维度爆炸",
            "通过主成分分析来增强数据特征的可解释性",
            "数据标准化通过标准化数据来减少噪声，并提高模型准确性"
        ],
        answer: "C",
        explanation: "PCA可能降低特征可解释性，因为它产生的是原始特征的线性组合。"
    },
    {
        id: 102,
        type: "single",
        question: "关于反向传播，以下描述错误的是哪一项？",
        options: [
            "反向传播可以结合梯度下降算法更新网络权重",
            "反向传播指的是误差通过网络反向传播",
            "反向传播只能在前馈神经网络中运用",
            "反向传播会使用激活函数的梯度"
        ],
        answer: "C",
        explanation: "反向传播也可用于其他类型神经网络如RNN、CNN等。"
    },
    {
        id: 103,
        type: "single",
        question: "输入一个 32x32 的图像，用大小为5x5 的卷积核进行步长为1的卷积计算，输出的图像尺寸为以下哪个选项？",
        options: [
            "23x23",
            "28x23",
            "28x28",
            "29x29"
        ],
        answer: "C",
        explanation: "输出尺寸 = (输入尺寸 - 卷积核大小 + 2*填充)/步长 + 1 = (32-5+0)/1+1 = 28"
    },
    {
        id: 104,
        type: "single",
        question: "异腾大模型解决方案提供了全流程的工具，以下哪个工具可用于性能调优？",
        options: [
            "MindIE",
            "MindSpore",
            "MindStudio",
            "MindSpeed"
        ],
        answer: "C",
        explanation: "MindStudio提供性能调优功能。"
    },
    {
        id: 112,
        type: "single",
        question: "深度学习常用的损失函数有均方误差和交叉熵误差，针对两者的使用场景，以下哪种描述是正确的？",
        options: [
            "交叉熵误差更多用于回归问题",
            "两者均可用于分类问题",
            "均方误差更多用于分类问题",
            "两者均可用于回归问题"
        ],
        answer: "B",
        explanation: "两者都可用于分类，但交叉熵更常见；均方误差主要用于回归。"
    },
    {
        id: 113,
        type: "single",
        question: "一张图片依次经过以下网络结构:Conv2d(3→16,3x3,1,1)→MaxPool2d(2x2,2)→Flatten→Linear(n,10)，输入32x32x3，则n等于？",
        options: [
            "3600",
            "4096",
            "3136",
            "4624"
        ],
        answer: "B",
        explanation: "卷积后尺寸不变(32x32)，池化后变为16x16，16x16x16=4096"
    },
    {
        id: 114,
        type: "single",
        question: "在 MindSpore 中，运行以下代码后，描述正确的是?a=([1,2],[3,4],[56,7])x= Tensor (a, dtype.Int32)",
        options: [
            "程序异常，抛出 TypeError",
            "程序异常，a必须转化为数组，才能正常执行",
            "创建一个维度为(3,2)的 tensor",
            "创建一个维度为(2,3)的 tensor"
        ],
        answer: "A",
        explanation: "列表维度不一致会导致TypeError。"
    },
    {
        id: 123,
        type: "single",
        question: "一张图片依次经过以下网络结构:conv1=nn.Conv2d(in channels=3,out channels=16,kernel size=7,stride=1,padding=1)pool= nn.MaxPool2d(kernel size=2, stride=2, padding=0) flatten=nn.Flatten()fc1=nn.Linear(n,10)假设输入图片的大小为 32x32x3，n的值应该设置为以下哪个选项？",
        options: [
            "4624",
            "136B",
            "4096",
            "3600"
        ],
        answer: "C",
        explanation: "卷积后尺寸=(32-7+2*1)/1+1=28，池化后=14，14x14x16=3136"
    },
    {
        id: 127,
        type: "single",
        question: "从有标签的历史数据中来预测下季度的商铺营收额，这是一个什么问题？",
        options: [
            "回归问题",
            "规则问题",
            "聚类问题",
            "分类问题"
        ],
        answer: "A",
        explanation: "预测连续数值是回归问题。"
    },
    {
        id: 129,
        type: "single",
        question: "机器学习算法中，以下哪一项不属于集成学习策略？",
        options: [
            "Bagging",
            "Marking",
            "Boosting",
            "Stacking"
        ],
        answer: "B",
        explanation: "Marking不是集成学习方法。"
    },
    {
        id: 131,
        type: "single",
        question: "智能驾驶的能力越来越强，以下哪一项因素不是支撑它发展的因素？",
        options: [
            "智能座舱语音助手",
            "优质的传感设备",
            "强大的AI算法",
            "高性能的车机芯片"
        ],
        answer: "A",
        explanation: "语音助手不是智能驾驶的核心支撑技术。"
    },
    {
        id: 132,
        type: "single",
        question: "某工程师在使用MindsDK构建视频分析AI应用时，可以选择工具中的零个模块？",
        options: [
            "Vision SDK",
            "RAG SDK",
            "Rec SDK",
            "lndex SDK"
        ],
        answer: "A",
        explanation: "Vision SDK用于视觉任务，适合视频分析。"
    },
    {
        id: 133,
        type: "single",
        question: "以下关于CPU低时延设计的描述，错误的是哪一个选项？",
        options: [
            "强大的ALU单元，可在很短时钟周期完成计算",
            "复杂逻辑控制单元，多分支程序可通过分支预测能力降低时延",
            "有很多ALU和很少Cache,缓存合并访问DRAM，降低时延",
            "高时钟频率降低时延"
        ],
        answer: "C",
        explanation: "CPU有大量Cache来降低访存延迟，不是很少Cache。"
    },
    {
        id: 134,
        type: "single",
        question: "以下不属于华为应用使能套件的是哪个选项？",
        options: [
            "MindSDK",
            "ModelZoo",
            "CANN",
            "MindIE"
        ],
        answer: "C",
        explanation: "CANN是计算架构，不是应用使能套件。"
    },
    {
        id: 142,
        type: "single",
        question: "Ascend Extension for PyTorch 是基于昇腾的深度学习适配框架，使昇腾 NPU 可以⽀持 PyTorch 框架，这种实现⽅式是以下哪个选项？",
        options: [
            "算⼦扩展",
            "原扩展",
            "新扩展",
            "⾃扩展"
        ],
        answer: "A",
        explanation: "算子扩展，答案是这么写的。"
    },
    {
        id: 143,
        type: "single",
        question: " deepseek R1-ZERO 使⽤了下哪项技术",
        options: [
            "RLHF",
            "GRPO",
            "DPO",
            "SFT"
        ],
        answer: "B",
        explanation: "GRPO，答案是这么写的。"
    },
    {
        id: 144,
        type: "single",
        question: " ohn 在 ai 应⽤开发过程中操作正确的以下哪个选项 ？",
        options: [
            "适当增⼤ batch size 提⾼GPU 的利⽤率",
            "加⼤epoch 提⾼训练精度",
            "可以适当删除部分训练参数",
            "训练完成后才保存参数 "
        ],
        answer: "A",
        explanation: "适当增⼤ batch size 提⾼GPU 的利⽤率，答案是这么写的。"
    },
    // 多选题 (29道)
    {
        id: 6,
        type: "multi",
        question: "以下哪些选项不能用于聚类任务？",
        options: [
            "DNN",
            "k-means",
            "DBSCAN",
            "PCA"
        ],
        answer: ["A", "D"],
        explanation: "DNN(深度神经网络)主要用于分类和回归任务，PCA(主成分分析)是降维方法，两者都不直接用于聚类。"
    },
    {
        id: 7,
        type: "multi",
        question: "LSTM 网络结构中由哪些模块共同决定了当前细胞状态需要保留的信息？",
        options: [
            "上一层的隐藏状态",
            "当前的输入信息",
            "当前的隐藏状态",
            "上一层的细胞状态"
        ],
        answer: ["A", "B", "D"],
        explanation: "LSTM中，遗忘门、输入门和输出门共同决定细胞状态，其中遗忘门由上一层的隐藏状态和当前输入决定，输入门决定哪些新信息加入细胞状态。"
    },
    {
        id: 9,
        type: "multi",
        question: "John 在使用 AI框架实现并行训练时选择了通过梯度累积的方式进行，以下关于他操作的描述中，错误的是哪几项？",
        options: [
            "为了保证训练结果与原批量训练可以进行对比，梯度累积一般不使用优化器参考",
            "梯度累积的实现，不需要进行任何代码的调整",
            "PyTorch、MindSpore等框架实现方式大致相同，都可以直接使用，不需要进行调整",
            "对于输入不稳定计算(如动态 batchsize)的场景，不适合使用梯度累积"
        ],
        answer: ["B", "C"],
        explanation: "梯度累积需要代码调整，不同框架实现方式有差异。"
    },
    {
        id: 10,
        type: "multi",
        question: "以下哪些选项属于终端设备上所使用的 AI芯片特征？",
        options: [
            "低成本",
            "高功耗",
            "高性能",
            "低延迟"
        ],
        answer: ["A", "C", "D"],
        explanation: "终端设备AI芯片需要低成本、高性能和低延迟，但通常要求低功耗而非高功耗。"
    },
    {
        id: 11,
        type: "multi",
        question: "与 ASIC 芯片相比，以下哪些选项是FPGA 芯片的特征？",
        options: [
            "实现相同功能时，需要的芯片面积更大",
            "相同工艺条件下，功耗更小",
            "运行时无需加载配置，可立即运行",
            "研发风险较低"
        ],
        answer: ["A", "D"],
        explanation: "FPGA芯片面积通常更大，研发风险较低，但功耗不一定更小，且运行时需要加载配置。"
    },
    {
        id: 27,
        type: "multi",
        question: "以下哪些选项属于图像识别技术应用场景？",
        options: [
            "无人驾驶",
            "安防监控",
            "智能机器人",
            "机器狗"
        ],
        answer: ["A", "B", "C", "D"],
        explanation: "所有这些场景都涉及图像识别技术的应用。"
    },
    {
        id: 28,
        type: "multi",
        question: "智能家居已经走入我们的生活，AI可以赋能哪些日常应用？",
        options: [
            "室内监控",
            "语音控制家电",
            "门锁",
            "室内灯光控制"
        ],
        answer: ["A", "B", "C", "D"],
        explanation: "AI在智能家居中广泛应用于这些场景。"
    },
    {
        id: 32,
        type: "multi",
        question: "以下哪些芯片可以用来训练AI模型？",
        options: [
            "CPU",
            "GPU",
            "XPU",
            "NPU"
        ],
        answer: ["A", "B", "D"],
        explanation: "CPU、GPU和NPU都可以用于AI模型训练，XPU不是通用术语。"
    },
    {
        id: 45,
        type: "multi",
        question: "在转换医疗场景下，以下技术与应用对药效的思考一致的是？",
        options: [
            "药物流程-语音合成",
            "虚拟助理-语音识别",
            "疾病风险预测-数据挖掘",
            "医学影像-计算机视觉"
        ],
        answer: ["C", "D"],
        explanation: "疾病风险预测和医学影像分析都涉及对医疗效果的分析。"
    },
    {
        id: 55,
        type: "multi",
        question: "只有更完善的基础数据服务产业，才是 AI技术能更好发展的基石。关于数据，以下哪些选项是正确的？",
        options: [
            "数据质量和数量对 AI 发展来说很重要",
            "消除数据孤岛现象对 AI 技术拓展很重要",
            "消除数据壁垒对 AI 技术发展很重要",
            "更安全的数据共享是 AI 技术更好发展的基石之一"
        ],
        answer: ["A", "B", "C", "D"],
        explanation: "所有这些因素都对AI数据生态很重要。"
    },
    {
        id: 56,
        type: "multi",
        question: "评判分类模型的 F1 值是以下哪些指标的调和均值？",
        options: [
            "精确度 (precision)",
            "准确率 (accuracy)",
            "有效率 (validity)",
            "召回率 (recall)"
        ],
        answer: ["A", "D"],
        explanation: "F1值是精确度和召回率的调和平均数。"
    },
    {
        id: 57,
        type: "multi",
        question: "以下关于决策树的描述中，哪些选项是错误的？",
        options: [
            "ID3,C4.5, CART 等决策树算法都可以用于分类问题",
            "ID3, C4.5, CART 等决策树算法都可以用于回归问题",
            "纯度的量化指标不包括方差",
            "决策树的构造就是进行属性的选择，确定各个特征属性之间的树结构"
        ],
        answer: ["B", "C"],
        explanation: "ID3和C4.5主要用于分类，CART可用于分类和回归；纯度指标包括方差(用于回归)。"
    },
    {
        id: 58,
        type: "multi",
        question: "机器学习中模型总会存在误差，误差包括哪些部分？",
        options: [
            "偏差",
            "数据异常值",
            "协方差",
            "方差"
        ],
        answer: ["A", "D"],
        explanation: "模型误差主要包括偏差和方差。"
    },
    {
        id: 59,
        type: "multi",
        question: "关于卷积神经网络池化层，以下哪些描述是正确的？",
        options: [
            "池化操作采用扫描窗口实现",
            "常用的池化方法有最大池化和平均池化",
            "池化层可以减少特征图的尺寸",
            "经过池化的特征图一定会变小"
        ],
        answer: ["A", "B", "C"],
        explanation: "池化通常减少特征图尺寸，但步长为1时尺寸可能不变。"
    },
    {
        id: 60,
        type: "multi",
        question: "如果深度神经网络出现了梯度消失或梯度爆炸问题，以下哪些选项是常用的缓解方式？",
        options: [
            "使用 Relu 激活函数",
            "随机欠采样",
            "正则化",
            "梯度剪切"
        ],
        answer: ["A", "D"],
        explanation: "ReLU缓解梯度消失，梯度剪切缓解梯度爆炸。"
    },
    {
        id: 61,
        type: "multi",
        question: "以下哪些结构属于 LSTM？",
        options: [
            "记忆门",
            "输出门",
            "输入门",
            "遗忘门"
        ],
        answer: ["B", "C", "D"],
        explanation: "LSTM包含输入门、遗忘门、输出门和记忆单元。"
    },
    {
        id: 62,
        type: "multi",
        question: "John 在使用 AI 框架构建并训练模型时，选择了通过静态图的方式进行，以下关于他操作的描述中，错误的是哪几项？",
        options: [
            "模型训练完成以后，可以进行跨平台的应用部署",
            "为了保证网络各层之间的维度可以进行计算，随时输出一些中间结果作为参考",
            "对于输入不确定性高 (如动态 batch size) 的任务，不适合使用静态图",
            "Pytorch、MindSpore 等框架默认为静态图模式，也可以直接使用，不需要做其他配置"
        ],
        answer: ["B", "D"],
        explanation: "静态图不适合随时输出中间结果；PyTorch默认是动态图。"
    },
    {
        id: 63,
        type: "multi",
        question: "以下对数据的操作中可以增加模型泛化能力的有哪些选项？",
        options: [
            "对图片数据进行裁剪、反转操作",
            "复制当前数据，增加数据量",
            "对文字类数据进行随机删除",
            "对音频类数据添加噪声"
        ],
        answer: ["A", "C", "D"],
        explanation: "数据增强可以提高泛化能力，但简单复制数据不会。"
    },
    {
        id: 64,
        type: "multi",
        question: "使用 AI 芯片训练模型时，对芯片以下哪些能力要求较高？",
        options: [
            "可扩展性",
            "芯片面积",
            "算力",
            "精度"
        ],
        answer: ["A", "C", "D"],
        explanation: "AI训练芯片需要高算力、高精度和良好可扩展性。"
    },
    {
        id: 65,
        type: "multi",
        question: "以下哪些属于机器学习中常用的特征工程操作？",
        options: [
            "特征归一化",
            "特征离散化",
            "特征交叉",
            "特征删除"
        ],
        answer: ["A", "B", "C"],
        explanation: "特征删除是特征选择，不是特征工程的核心操作。"
    },
    {
        id: 66,
        type: "multi",
        question: "以下哪些选项是构建知识图谱的主要步骤？",
        options: [
            "确定领域",
            "信息抽取",
            "获取数据",
            "数据加密"
        ],
        answer: ["A", "B", "C"],
        explanation: "数据加密不是构建知识图谱的核心步骤。"
    },
    {
        id: 75,
        type: "multi",
        question: "以下哪个选项是归并排序的基本思想？",
        options: [
            "找一个较小的数，把所有的数和这个数比较把小的移到小的序列上",
            "把它分成两个个数大致相同的子数组，对这两个子数组分别采用归并排序的方法排序",
            "得到原序列之后，选一个枢轴，把小于这个枢轴的放左边，大于一个枢轴的放右边",
            "把它分成两个个数大致相同的子数组，然后再把这两个子数组合并"
        ],
        answer: ["B", "D"],
        explanation: "归并排序采用分治思想：分割数组→分别排序→合并结果。"
    },
    {
        id: 86,
        type: "multi",
        question: "以下哪些选项不是决策树用于划分节点的依据？",
        options: [
            "信息增益率",
            "频率",
            "方差",
            "期望"
        ],
        answer: ["B", "D"],
        explanation: "决策树使用信息增益、增益率、基尼指数或方差等，不使用频率和期望。"
    },
    {
        id: 87,
        type: "multi",
        question: "以下关于大模型和小模型的对比描述正确的有哪些选项？",
        options: [
            "小模型在部署和使用上更加灵活",
            "行业大模型通常使用微调的方式训练",
            "使用大模型的数据集训练小模型，小模型也会产生涌现",
            "小模型可以使用思维链技术来提升准确度"
        ],
        answer: ["A", "B"],
        explanation: "小模型通常不会出现涌现现象，思维链主要对大模型有效。"
    },
    {
        id: 88,
        type: "multi",
        question: "在界腾 AI 处理器中，AI Core 的存储控制单元可以完成以下哪些操作？",
        options: [
            "转置",
            "补零",
            "解压缩",
            "Img2Col"
        ],
        answer: ["A", "B", "C", "D"],
        explanation: "AI Core的存储控制单元支持这些数据预处理操作。"
    },
    {
        id: 89,
        type: "multi",
        question: "界腾 AI 处理器包含 AI Core 模块，这个模块包含以下哪些选项？",
        options: [
            "矩阵计算单元",
            "存储控制单元",
            "指令发射模块",
            "寄存器"
        ],
        answer: ["A", "B", "C", "D"],
        explanation: "AI Core包含所有这些组件。"
    },
    {
        id: 90,
        type: "multi",
        question: "以下哪些属于多模态模型的输入类型？",
        options: [
            "文本",
            "图像",
            "音频",
            "视频"
        ],
        answer: ["A", "B", "C", "D"],
        explanation: "多模态模型可以处理所有这些类型的数据。"
    },
    {
        id: 91,
        type: "multi",
        question: "当使用机器学习建立模型的过程中，以下哪些属于必备的操作？",
        options: [
            "特征选择",
            "数据获取",
            "超参数调节",
            "模型构建"
        ],
        answer: ["A", "D"],
        explanation: "特征选择和模型构建是必备步骤，数据获取和超参数调节虽重要但非绝对必需。"
    },
    {
        id: 92,
        type: "multi",
        question: "深度学习中，数据增强是一种有效防止过拟合的方法。以下关于数据增强的描述中，正确的有哪几项？",
        options: [
            "在图像分类任务中对图片进行旋转、缩放",
            "语音识别任务中对输入数据添加随机噪声",
            "NLP中复制同一个词多次进行训练",
            "在图像分类任务中调节图片亮度或对比度"
        ],
        answer: ["A", "B", "D"],
        explanation: "简单复制词语不是有效的数据增强方法。"
    },
    {
        id: 105,
        type: "multi",
        question: "某工程师使用一个全连接神经网络实现了 10 分类任务，此网络有2层隐藏层，每层均有 32 个神经元，输出层有10 个神经元，输入层有 20 个神经元。现在需要修改此网络结构以实现 2分类任务，以下哪些修改不能实现该功能？",
        options: [
            "增加一层隐藏层，神经元数为 16，并将输出层神经元数量修改为 2",
            "将隐藏层和输出层神经元数量减半",
            "减少输入层神经元数量",
            "减少一层隐藏层，将剩下的隐藏层神经元数量减半"
        ],
        answer: ["B", "C", "D"],
        explanation: "只有A选项正确修改了输出层为2个神经元以适应二分类。"
    },
    {
        id: 106,
        type: "multi",
        question: "John 想要使用 PyTorch 实现一个用于机器翻译的模型，以下关于他操作描述正确的有哪些选项？",
        options: [
            "重载__init__方法用于模型初始化",
            "重载backward 方法用于定义自动微分过程",
            "重载construct 方法用于定义前向计算过程",
            "构建的模型类应该继承 nn.Module"
        ],
        answer: ["A", "D"],
        explanation: "PyTorch中不需要重载backward和construct方法。"
    },
    {
        id: 115,
        type: "multi",
        question: "以下描述中不属于包装法(Wrapper)局限的是哪些选项？",
        options: [
            "倾向于选择冗余的变量，因为没有考虑特征之间的关系",
            "包装法是预测模型用于评估特征组合的工具，要根据模型的准确性进行评分",
            "因为包装法为每个子集训练一个新模型，所以它们的计算量非常大",
            "其特征选择的方法通常为特定类型的模型提供了性能最好的特性集"
        ],
        answer: ["B", "D"],
        explanation: "B和D描述的是包装法的优点而非局限。"
    },
    {
        id: 116,
        type: "multi",
        question: "以下哪些选项属于常见的脏数据？",
        options: [
            "格式错误的值",
            "缺失值",
            "重复值",
            "逻辑错误的值"
        ],
        answer: ["A", "B", "C", "D"],
        explanation: "所有这些都属于脏数据的常见类型。"
    },
    {
        id: 117,
        type: "multi",
        question: "企业在实例分割项目中(MindSpore)将图片转为 MindRecord 格式，以下正确描述有哪些？",
        options: [
            "实现数据统一存储、访问，方便数据管理",
            "有效压缩数据体积，方便数据存储率",
            "提升模型的训练计算速度，缩短开发周期",
            "减少磁盘 10，网络 10 开销，缩短开发周期"
        ],
        answer: ["A", "B", "D"],
        explanation: "MindRecord格式主要优化存储和IO，不直接提升训练计算速度。"
    },
    {
        id: 118,
        type: "multi",
        question: "以下哪些选项属于 AI开发框架提供的功能？",
        options: [
            "跨平台分布式训练",
            "数据预处理",
            "算子 API",
            "自动模型调优"
        ],
        answer: ["A", "B", "D"],
        explanation: "算子API是框架基础功能，但不是所有框架都提供自动模型调优。"
    },
    {
        id: 124,
        type: "multi",
        question: "深度学习是比较火热的人工智能技术，但是在做深度学习任务时常会遇到各种各样的问题，以下哪些问题会在深度学习任务中出现？",
        options: [
            "梯度爆炸问题",
            "梯度消失问题",
            "过拟合问题",
            "数据不平衡问题"
        ],
        answer: ["A", "B", "C", "D"],
        explanation: "深度学习中常见这些问题。"
    },
    {
        id: 128,
        type: "multi",
        question: "John 使用 AI框架创建了一个 tensorA，A具备以下哪些属性？",
        options: [
            "数据类型",
            "维数",
            "形状",
            "存储位置"
        ],
        answer: ["A", "B", "C", "D"],
        explanation: "张量包含所有这些属性。"
    },
    {
        id: 130,
        type: "multi",
        question: "以下关于 AI应用开发流程的描述错误的有哪些选项？",
        options: [
            "构建网络模型时需要结合具体需求，考虑损失函数、各层数神经元数量",
            "模型训练过程中，为保证拟合效果要尽可能的增加训练时长",
            "模型训练完成后，只将超参数保存为模型文件方便后续调用",
            "数据收集阶段要尽量获取更多的数据，数据的数量比质量更重要"
        ],
        answer: ["B", "C", "D"],
        explanation: "这些描述都存在错误：训练时间过长可能过拟合；需要保存模型权重而不仅是超参数；数据质量比数量更重要。"
    },
    {
        id: 135,
        type: "multi",
        question: "某工程师想要部署DeepSeek V3模型进行推理，他需要考虑以下哪些问题？",
        options: [
            "模型结构",
            "设备显示/内存容量",
            "是否使用量化/量化方式",
            "微调数据"
        ],
        answer: ["A", "B", "C"],
        explanation: "推理时不需要微调数据。"
    },
    {
        id: 136,
        type: "multi",
        question: "使用AI芯片训练模型时，对芯片以下哪些能力要求较高？",
        options: [
            "芯片面积",
            "算力",
            "精度",
            "可扩展性"
        ],
        answer: ["B","C", "D"],
        explanation: "训练芯片需要高精度、算力和良好可扩展性。"
    },
    {
        id: 145,
        type: "multi",
        question: "A 企业在某个实际的项⽬中（基于 MindSpore 实现）使⽤了⼤量的图⽚数据，为了节省空间，AI⼯程师将数据转化为了 MindRecord 格式，以下关于该描述正确的说法有哪些？",
        options: [
            "转化为 MindRecord 后，其数据统⼀存储、访问，⽅便数据管理",
            "转化为 MindRecord 后，可以提升数据的访问速度，缩短开发周期",
            "转化为 MindRecord 后，减少磁盘 I/O、⽹络 I/O 开销，缩短开发",
            "转化为 MindRecord 后，可以有效进⾏数据保护，⽅便数据存"
        ],
        answer: ["A", "B", "C"],
        explanation: "ABC，答案是这么写的。"
    },
    {
        id: 146,
        type: "multi",
        question: "MindSpore 中，mindspore.nn.Conv2d 构建卷积层时，可能会使⽤到以下哪⼏个参数？",
        options: [
            "kernel_size",
            "bias",
            "inbias",
            "padding"
        ],
        answer: ["A", "B", "D"],
        explanation: "ABD，答案是这么写的。"
    },
    {
        id: 147,
        type: "multi",
        question: "昇腾 ai 处理器包含 ai core 模块，这个模块包含以下哪些选项？",
        options: [
            "存储缓存单元",
            "指令发射模块",
            "存储单元",
            "矩阵计算单"
        ],
        answer: ["A", "B", "C", "D"],
        explanation: "ABCD，答案是这么写的。"
    },
    // 判断题 (64道)
    {
        id: 12,
        type: "judgment",
        question: "DeepSeek V3 是一个基于 GPT 结构的多模态模型，较 V2 版本相比，其训练成本和推理速度都有很大的提升。",
        options: [
            "正确",
            "错误"
        ],
        answer: "错误",
        explanation: "DeepSeek V3相比V2版本在训练成本和推理速度上并没有很大的提升。"
    },
    {
        id: 13,
        type: "judgment",
        question: "在随机森林中，基学习器所使用的特征衡量指标需要是基尼系数。",
        options: [
            "正确",
            "错误"
        ],
        answer: "错误",
        explanation: "随机森林的基学习器可以使用基尼系数或信息增益等不同指标。"
    },
    {
        id: 14,
        type: "judgment",
        question: "损失函数与激活函数是一类函数",
        options: [
            "正确",
            "错误"
        ],
        answer: "错误",
        explanation: "损失函数用于衡量预测值与真实值的差距，激活函数用于引入非线性，两者功能不同。"
    },
    {
        id: 15,
        type: "judgment",
        question: "Self-Attention机制无法捕捉到序列的位置信息，因此在 Transformer 模型中需要添加位置编码。",
        options: [
            "正确",
            "错误"
        ],
        answer: "正确",
        explanation: "Self-Attention本身不包含位置信息，需要位置编码来提供序列顺序信息。"
    },
    {
        id: 16,
        type: "judgment",
        question: "MoE 是一种混合专家模型，这种方案可以有效增强模型的容量和能力，同时在推理时能够动态决定哪些专家模型工作",
        options: [
            "正确",
            "错误"
        ],
        answer: "正确",
        explanation: "MoE(混合专家)模型通过多个专家网络和门控机制动态选择专家，增强模型能力。"
    },
    {
        id: 17,
        type: "judgment",
        question: "在 RNN 中，每一时刻隐藏状态的权重都保持一致，这样可以减少网络的参数量。",
        options: [
            "正确",
            "错误"
        ],
        answer: "正确",
        explanation: "RNN在不同时间步共享权重，减少了参数量。"
    },
    {
        id: 18,
        type: "judgment",
        question: "AI开发框架的功能是对开发者提供网络模型接口，模型训练过程中的前向传播和反向传播需要开发者自己完成。",
        options: [
            "正确",
            "错误"
        ],
        answer: "错误",
        explanation: "现代AI框架通常自动处理前向传播和反向传播，开发者只需定义网络结构。"
    },
    {
        id: 19,
        type: "judgment",
        question: "MindSpore 在构建神经网络时，可以通过继承 Cell 类，并复-lnit-方法和construct 方法实现。",
        options: [
            "正确",
            "错误"
        ],
        answer: "正确",
        explanation: "MindSpore中通过继承Cell类并实现__init__和construct方法来构建神经网络。"
    },
    {
        id: 21,
        type: "judgment",
        question: "与 FPGA相比，相同工艺下，基于 ASIC 的 Ali芯片功耗更大，因此不适合用于手机等终端设备上。",
        options: [
            "正确",
            "错误"
        ],
        answer: "错误",
        explanation: "ASIC芯片通常比FPGA功耗更低，更适合终端设备。"
    },
    {
        id: 22,
        type: "judgment",
        question: "CPU 主要提供通用计算，适合复杂逻辑运算，70% 以上的晶体管用于构建算数逻辑单元、缓存 Cache 和控制单元。",
        options: [
            "正确",
            "错误"
        ],
        answer: "错误",
        explanation: "CPU中大部分晶体管用于缓存和控制逻辑，ALU只占一小部分。"
    },
    {
        id: 23,
        type: "judgment",
        question: "随着大模型技术发展，大语言模型已经赋能机器人领域，它使机器人不仅能听懂人类指令，还能认识三维世界中的物体。",
        options: [
            "正确",
            "错误"
        ],
        answer: "正确",
        explanation: "大语言模型确实在机器人领域有应用，帮助理解和执行指令。"
    },
    {
        id: 24,
        type: "judgment",
        question: "理论上，训练时的精度会随着模型复杂度的上升和训练时间的增加不断减小。",
        options: [
            "正确",
            "错误"
        ],
        answer: "错误",
        explanation: "训练精度通常随模型复杂度和训练时间增加而提高，但可能出现过拟合。"
    },
    {
        id: 33,
        type: "judgment",
        question: "John 在使用 PyTorch 框架实现 AI模型训练时，他应该先通过 set context设置模型训练相关参数，如设备ID、计算图模式等",
        options: [
            "正确",
            "错误"
        ],
        answer: "错误",
        explanation: "set context是MindSpore的API，不是PyTorch的。"
    },
    {
        id: 67,
        type: "judgment",
        question: "卷积神经网络中，卷积核的大小通常为奇数。",
        options: [
            "正确",
            "错误"
        ],
        answer: "正确",
        explanation: "奇数大小的卷积核便于对称填充和处理。"
    },
    {
        id: 68,
        type: "judgment",
        question: "卷积神经网络中的卷积层可以实现参数共享，减少网络参数。参数共享是指在同一层中使用的全部卷积核参数一致，不同层使用的卷积核参数是不同的。",
        options: [
            "正确",
            "错误"
        ],
        answer: "正确",
        explanation: "卷积层的参数共享指同一卷积核在输入的不同位置共享参数。"
    },
    {
        id: 69,
        type: "judgment",
        question: "随机森林是一种集成学习算法，通过多个决策树的投票得到最终结果。",
        options: [
            "正确",
            "错误"
        ],
        answer: "正确",
        explanation: "随机森林通过多个决策树集成，投票决定最终结果。"
    },
    {
        id: 70,
        type: "judgment",
        question: "当前 AI 开发框架动态计算图和静态计算图语法兼容，可以直接进行相互转换。",
        options: [
            "正确",
            "错误"
        ],
        answer: "错误",
        explanation: "动态图和静态图语法不完全兼容，不能直接转换。"
    },
    {
        id: 71,
        type: "judgment",
        question: "ReLU 激活函数在 x>0 时导数为 1，可缓解梯度消失问题。",
        options: [
            "正确",
            "错误"
        ],
        answer: "正确",
        explanation: "ReLU在正区间梯度为1，缓解了梯度消失问题。"
    },
    {
        id: 72,
        type: "judgment",
        question: "AscendCL 可以基于第三方框架开发推理类应用，但是需要通过 ATC 工具先对模型进行转换。",
        options: [
            "正确",
            "错误"
        ],
        answer: "正确",
        explanation: "AscendCL开发推理应用需要先用ATC工具转换模型。"
    },
    {
        id: 78,
        type: "judgment",
        question: "Seq2Seq 模型常用于机器翻译、文本摘要等序列生成任务。",
        options: [
            "正确",
            "错误"
        ],
        answer: "正确",
        explanation: "Seq2Seq模型确实用于序列到序列的任务如翻译和摘要。"
    },
    {
        id: 79,
        type: "judgment",
        question: "量化是一种模型压缩技术，可以减少模型存储占用和推理延迟。",
        options: [
            "正确",
            "错误"
        ],
        answer: "正确",
        explanation: "量化通过降低数值精度来压缩模型和加速推理。"
    },
    {
        id: 93,
        type: "judgment",
        question: "异腾 AI 处理器只提供了 DDR 接口，可以外接 DDR4 内存。",
        options: [
            "正确",
            "错误"
        ],
        answer: "错误",
        explanation: "异腾AI处理器提供多种接口，不只是DDR。"
    },
    {
        id: 94,
        type: "judgment",
        question: "SVM、k-NN 和 k-means 都属于非参数模型。",
        options: [
            "正确",
            "错误"
        ],
        answer: "错误",
        explanation: "SVM是参数模型，k-NN和k-means是非参数模型。"
    },
    {
        id: 95,
        type: "judgment",
        question: "数据降维的目的是减少数据量，增加模型的准确度。",
        options: [
            "正确",
            "错误"
        ],
        answer: "错误",
        explanation: "降维可能损失信息，不一定增加准确度。"
    },
    {
        id: 96,
        type: "judgment",
        question: "Sigmoid、tanh和 Softsign 这些激活函数在网络层数加深时，都不能避免梯度消失的问题",
        options: [
            "正确",
            "错误"
        ],
        answer: "正确",
        explanation: "这些饱和激活函数在深层网络中容易出现梯度消失。"
    },
    {
        id: 97,
        type: "judgment",
        question: "机器学习是深度学习的一部分，人工智能也是深度学习的一部分。",
        options: [
            "正确",
            "错误"
        ],
        answer: "错误",
        explanation: "深度学习是机器学习的一部分，机器学习是人工智能的一部分。"
    },
    {
        id: 98,
        type: "judgment",
        question: "前馈神经网络接收前一层的输出，并输出给下一层，采用一种单向多层结构，每一层包含若干个神经元，同一层的神经元之间没有互相连接，层间信息的传送只沿一个方向进行。",
        options: [
            "正确",
            "错误"
        ],
        answer: "正确",
        explanation: "这是前馈神经网络的标准定义。"
    },
    {
        id: 107,
        type: "judgment",
        question: "大的模型在工业应用时会有运行效率的问题，所以为了保证效率，模型越小越好。",
        options: [
            "正确",
            "错误"
        ],
        answer: "错误",
        explanation: "模型大小需要在效果和效率间平衡，不是越小越好。"
    },
    {
        id: 108,
        type: "judgment",
        question: "朴素贝叶斯算法假设样本特征之间相互独立，且对于样本缺失值较为敏感。",
        options: [
            "正确",
            "错误"
        ],
        answer: "错误",
        explanation: "朴素贝叶斯对缺失值不敏感。"
    },
    {
        id: 109,
        type: "judgment",
        question: "PyTorch、MindSpore 等AI开发框架通常提供内置的预处理功能，用于数据采集、数据清洗、特征选择和归一化等操作。",
        options: [
            "正确",
            "错误"
        ],
        answer: "正确",
        explanation: "现代AI框架通常提供数据预处理功能。"
    },
    {
        id: 110,
        type: "judgment",
        question: "同一个模型，训练比推理要求的硬件性能更高。",
        options: [
            "正确",
            "错误"
        ],
        answer: "正确",
        explanation: "训练过程需要反向传播和参数更新，计算量更大。"
    },
    {
        id: 111,
        type: "judgment",
        question: "CANN 中提供了 DVPP和AIPP，前者使用异腾 AI处理器中的 DVPP 模块对图像进行处理，后者使用 AlCore 对图像进行处理。",
        options: [
            "正确",
            "错误"
        ],
        answer: "正确",
        explanation: "DVPP和AIPP是CANN中的图像处理模块。"
    },
    {
        id: 119,
        type: "judgment",
        question: "A模型的智能程度取决于输入数据的质量，算法在其中的作用可以忽略。",
        options: [
            "正确",
            "错误"
        ],
        answer: "错误",
        explanation: "算法和模型结构对智能程度同样重要。"
    },
    {
        id: 120,
        type: "judgment",
        question: "Self-Attention和 RNN 功能类似似可以处理时序相关问题，同时它们计算时的并行度相对都不高。",
        options: [
            "正确",
            "错误"
        ],
        answer: "错误",
        explanation: "Self-Attention的并行度远高于RNN。"
    },
    {
        id: 121,
        type: "judgment",
        question: "为防止过拟合，在训练过程中，可以插入对验证集数据的测试。当发现验证集数据的 Loss 上升时，提前停止训练。",
        options: [
            "正确",
            "错误"
        ],
        answer: "正确",
        explanation: "这是早停法的标准做法。"
    },
    {
        id: 122,
        type: "judgment",
        question: "张量无法进行加减乘除等算术运算。",
        options: [
            "正确",
            "错误"
        ],
        answer: "错误",
        explanation: "张量支持各种算术运算。"
    },
    {
        id: 125,
        type: "judgment",
        question: "DeepSeek-R1-Zero是DeepSeek 系列模型中体积最大的，它不需要通过强化学习训练，也能和人类偏好对齐。",
        options: [
            "正确",
            "错误"
        ],
        answer: "错误",
        explanation: "大模型通常需要RLHF来对齐人类偏好。"
    },
    {
        id: 126,
        type: "judgment",
        question: "AJ开发框架可以帮助开发者快速构建自己的模型，并且可以实现跨平台和跨硬件的分布式训练。",
        options: [
            "正确",
            "错误"
        ],
        answer: "正确",
        explanation: "现代AI框架支持跨平台和分布式训练。"
    },
    {
        id: 137,
        type: "judgment",
        question: "异常数据在模型训练过程中不需要删除，大量的异常数据可以提升模型的泛化能力。",
        options: [
            "正确",
            "错误"
        ],
        answer: "错误",
        explanation: "异常数据通常需要处理，否则可能损害模型性能。"
    },
    {
        id: 138,
        type: "judgment",
        question: "AI芯片会针对矩阵运算做加速设计。",
        options: [
            "正确",
            "错误"
        ],
        answer: "正确",
        explanation: "AI芯片专门优化矩阵运算。"
    },
    {
        id: 140,
        type: "judgment",
        question: "CPU可以通过提高频率的方式提升AI计算性能",
        options: [
            "正确",
            "错误"
        ],
        answer: "错误",
        explanation: "提高频率有物理限制，不是提升AI性能的主要方式。"
    },
    {
        id: 141,
        type: "judgment",
        question: "Self-Attention和RNN功能类似，可以处理时序相关问题，同时它们计算时的并行度相对都不高",
        options: [
            "正确",
            "错误"
        ],
        answer: "错误",
        explanation: "Self-Attention的并行度远高于RNN。"
    },
    {
        id: 148,
        type: "judgment",
        question: "MindSpore 在构建神经⽹络时，可以通过继承 Cell 类，并重写 init ⽅法和construct ⽅法实现",
        options: [
            "正确",
            "错误"
        ],
        answer: "正确",
        explanation: "见答案。"
    },
    {
        id: 149,
        type: "judgment",
        question: "RNN 与前馈神经⽹络最⼤的区别在于：每次都会将前时刻的计算结果（隐状态）带到下时刻的计算过程中，起训练",
        options: [
            "正确",
            "错误"
        ],
        answer: "正确",
        explanation: "见答案。"
    }
];
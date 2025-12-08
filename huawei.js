var single_choice_questions  = [
        {"type": "single_choice",
            "id": 1,
            "question": "以下哪个选项是昇腾针对AI场景推出的异构计算架构?",
            "options": ["A、CUDA", "B、Atlas", "C、CANN", "D、TBE"],
            "correct_answer": "C"
        },
        {"type": "single_choice",
            "id": 2,
            "question": "以下哪个选项是智算的特点?",
            "options": ["A、高精度", "B、低功耗", "C、大集群", "D、串行计算"],
            "correct_answer": "C"
        },
        {"type": "single_choice",
            "id": 3,
            "question": "关于使用Ascend C开发自定义算子的描述，以下哪个选项是错误的?",
            "options": [
                "A、自动并行调度，获得最优执行性能",
                "B、简单易用，遵循Python开发规则",
                "C、结构化核函数编程，简化算子开发逻辑",
                "D、CPU/NPU孪生调试，提升算子调试效率"
            ],
            "correct_answer": "B"
        },
        {"type": "single_choice",
            "id": 4,
            "question": "某工程师在Atlas系列硬件上训练模型时，想要优化各个计算节点的通信时长，可以选择CANN的哪个组件?",
            "options": ["A、GE", "B、AOL", "C、HCCL", "D、MindIE"],
            "correct_answer": "C"
        },
        {"type": "single_choice",
            "id": 5,
            "question": "根据AI终端白皮书，以下哪个选项是L4级AI的描述?",
            "options": [
                "A、AI作为工具被调用",
                "B、AI执行被分解的任务",
                "C、AI自主拆解及分配任务，闭环执行",
                "D、AI提供达到人类专家水平的定制化服务"
            ],
            "correct_answer": "D"
        },
        {"type": "single_choice",
            "id": 6,
            "question": "AskO3-X语音助手基于以下哪种技术实现全双工的通信?",
            "options": ["A、5G通信技术", "B、AI语音架构", "C、云计算技术", "D、物联网技术"],
            "correct_answer": "B"
        },
        {"type": "single_choice",
            "id": 7,
            "question": "某工程师想要实现一个深度学习算法，可以使用以下哪个框架来实现?",
            "options": ["A、MindSpore", "B、MindIE", "C、GPU", "D、HCCL"],
            "correct_answer": "A"
        },
        {"type": "single_choice",
            "id": 8,
            "question": "以下哪个选项不属于AI开发框架?",
            "options": ["A、MindSpore", "B、PyTorch", "C、TensorFlow", "D、Python"],
            "correct_answer": "D"
        },
        {"type": "single_choice",
            "id": 9,
            "question": "AI助力金融实现智慧运营，以下哪一项不属于智慧运营的能力?",
            "options": ["A、OCR智能识别", "B、智能客服助手", "C、基于财务信息和历史行为信息的信用评分", "D、文档助手"],
            "correct_answer": "C"
        },
        {"type": "single_choice",
            "id": 10,
            "question": "华为AI助力某证券交易所实现智能摘要，以下哪一项不属于智能摘要的功能?",
            "options": ["A、会议纪要", "B、通话总结", "C、企业年报摘要", "D、信贷报告生成"],
            "correct_answer": "D"
        },
        {"type": "single_choice",
            "id": 11,
            "question": "以下哪个选项是昇腾针对AI场景推出的异构计算架构?",
            "options": ["A、CUDA", "B、Atlas", "C、CANN", "D、TBE"],
            "correct_answer": "C"
        },
        {"type": "single_choice",
            "id": 12,
            "question": "以下哪个选项不属于数据预处理?",
            "options": ["A、格式标准化", "B、异常数据清理", "C、重复数据清除", "D、模型权重文件读取"],
            "correct_answer": "D"
        },
        {"type": "single_choice",
            "id": 13,
            "question": "以下哪个选项不属于CANN提供的工具链?",
            "options": ["A、Ascend C", "B、MindSpore", "C、GE", "D、HCCL"],
            "correct_answer": "B"
        },
        {"type": "single_choice",
            "id": 14,
            "question": "以下哪一款是华为自研的AI芯片?",
            "options": ["A、GTX4090", "B、昇腾910", "C、RX 9070XT", "D、H200"],
            "correct_answer": "B"
        },
        {"type": "single_choice",
            "id": 15,
            "question": "关于智算中心和云数据中心，以下哪个选项说法是错误的?",
            "options": [
                "A、智算中心芯片以CPU为主",
                "B、智算中心面向AI典型应用场景，如智能制造、智慧农业等",
                "C、云数据中心提供混合精度的通用算力",
                "D、云数据中心架构统一标准，互联互通互操作方便"
            ],
            "correct_answer": "A"
        },
        {"type": "single_choice",
            "id": 16,
            "question": "某工程师在使用MindsDK构建视频分析AI应用时，可以选择工具中的哪个模块?",
            "options": ["A、Index SDK", "B、RAG SDK", "C、Rec SDK", "D、Vision SDK"],
            "correct_answer": "D"
        },
        {"type": "single_choice",
            "id": 17,
            "question": "以下哪个选项是AI模型训练对存储诸多诉求的最终目的?",
            "options": ["A、提升GPU/NPU利用率", "B、提升模型准确率", "C、减少模型训练对设备数量的要求", "D、减少模型CKPT文件体积"],
            "correct_answer": "A"
        },
        {"type": "single_choice",
            "id": 18,
            "question": "以下不属于华为应用使能套件的是哪个选项?",
            "options": ["A、MindSDK", "B、ModelZoo", "C、CANN", "D、MindIE"],
            "correct_answer": "C"
        },
        {"type": "single_choice",
            "id": 19,
            "question": "当朋友发来一张图片，不知道图片里面是什么物体时，可以使用小艺以下哪个功能?",
            "options": ["A、识屏对话", "B、自动填表", "C、AI修图", "D、小艺通话"],
            "correct_answer": "A"
        },
        {"type": "single_choice",
            "id": 20,
            "question": "某工程师在训练模型时，可以通过以下哪个工具管理团队的NPU算力集群?",
            "options": ["A、Mind Edge", "B、MindCluster", "C、MindSpeed", "D、MindSpore"],
            "correct_answer": "B"
        },
        {"type": "single_choice",
            "id": 21,
            "question": "以下哪一项是AI4Science在宏观尺度上的应用?",
            "options": ["A、设计新的药物小分子", "B、设计蛋白质大分子", "C、设计新的材料晶型", "D、设计高效的飞行器"],
            "correct_answer": "D"
        },
        {"type": "single_choice",
            "id": 22,
            "question": "华为银行AI智算系统的全栈架构中，模型层实现以下哪一项功能?",
            "options": [
                "A、围绕新一代AI基础模型构建可消费、可调用、可迭代的AI能力，模型可编排、可调整、可组合",
                "B、围绕算力底座的资源调度和管理，以及AI模型能力的生产部署，使能算力、数据、算法资源可用",
                "C、以计算、网络、存储等产品为核心，形成标准化、可交付、可持续运营的算力基础设施",
                "D、结合自身场景应用，使用AI能力改进业务，让AI算力、算法、数据产生战略价值、商业价值、业务价值"
            ],
            "correct_answer": "A"
        },
        {"type": "single_choice",
            "id": 23,
            "question": "关于AI技术，以下哪项描述是错误的?",
            "options": [
                "A、AI模型是从数据中学习，不同类型的任务需要的训练数据也不完全相同",
                "B、训练AI模型使用的的数据类别可以是图片，文字或者语音",
                "C、训练AI模型可以不使用数据和AI计算芯片",
                "D、AI框架可以提高工程师实现AI模型的效率"
            ],
            "correct_answer": "C"
        },
        {"type": "single_choice",
            "id": 24,
            "question": "某工程师希望通过AI模型实现水果图像分类，以下哪项是训练模型时需要使用的数据?",
            "options": ["A、水果甜度", "B、水果图片", "C、水果颜色值", "D、敲击水果声音数据"],
            "correct_answer": "B"
        },
        {"type": "single_choice",
            "id": 25,
            "question": "以下哪个选项不是智算设备大内存(显存)的作用?",
            "options": ["A、减少模型训练时间", "B、降低模型对算力卡数量的要求", "C、提高模型的准确率", "D、增加模型吞吐量"],
            "correct_answer": "C"
        }
    ];

    var multi_choice_questions = [
        {"type": "multi_choice",
            "id": 1001,
            "question": "以下哪些选项是华为昇腾硬件产品?",
            "options": ["A、Atlas 800I A2", "B、Atlas 300I Pro", "C、Atlas 900PoD", "D、H100"],
            "correct_answer": ["A", "B", "C"]
        },
        {"type": "multi_choice",
            "id": 1002,
            "question": "随着AI发展，其产业链也越发完善。以下哪些选项属于AI的产业链?",
            "options": ["A、昇腾服务器", "B、MindSpore框架", "C、AI应用", "D、生成式模型算法"],
            "correct_answer": ["A", "B", "C", "D"]
        },
        {"type": "multi_choice",
            "id": 1003,
            "question": "以下哪些选项是智算集群需要高速无损网络的原因?",
            "options": [
                "A、AI计算的迭代过程中，会有大量突发流量",
                "B、AI计算中如果有流发生延迟，会导致存储和计算资源无法充分利用",
                "C、AI计算中会有不间断的小流量产生",
                "D、AI计算发生丢包中断，只能从头开始训练"
            ],
            "correct_answer": ["A", "B"]
        },
        {"type": "multi_choice",
            "id": 1004,
            "question": "某工程师在使用ModelZoo时，可以选择以下哪些类型的模型?",
            "options": ["A、多模态大模型", "B、大语言模型", "C、集成学习模型", "D、视觉模型"],
            "correct_answer": ["A", "B", "D"]
        },
        {"type": "multi_choice",
            "id": 1005,
            "question": "AI4Science在以下哪些选项中已经有了实际的应用?",
            "options": ["A、材料", "B、生命科学", "C、流体", "D、电磁"],
            "correct_answer": ["A", "B", "C", "D"]
        },
        {"type": "multi_choice",
            "id": 1006,
            "question": "为了支撑大模型高效训练，存储设备应该具备以下哪些特性?",
            "options": ["A、高IO", "B、多协议融合互通", "C、高带宽", "D、低功耗"],
            "correct_answer": ["A", "B", "C"]
        },
        {"type": "multi_choice",
            "id": 1007,
            "question": "AskO3接入Deepseek-R1后，产生以下哪些优化?",
            "options": ["A、增强深度推理能力", "B、完全替代人工决策", "C、支持多Agent复杂问题推演", "D、仅用于文本搜索"],
            "correct_answer": ["A", "C"]
        },
        {"type": "multi_choice",
            "id": 1008,
            "question": "训练大语言模型的关键设备包括以下哪些选项?",
            "options": ["A、算力设备", "B、存储设备", "C、网络设备", "D、基站设备"],
            "correct_answer": ["A", "B", "C"]
        },
        {"type": "multi_choice",
            "id": 1009,
            "question": "以下哪些选项是智算芯片面临的挑战?",
            "options": ["A、高算力", "B、大带宽", "C、高功耗", "D、高能效"],
            "correct_answer": ["A", "B"]
        },
        {"type": "multi_choice",
            "id": 1010,
            "question": "以下哪些选项属于Mindspore框架的优点?",
            "options": ["A、手动微分", "B、易用", "C、高效", "D、全场景"],
            "correct_answer": ["B", "C", "D"]
        },
        {"type": "multi_choice",
            "id": 1011,
            "question": "以下关于某工程师在使用MindSDK时的操作，描述正确的有哪些选项?",
            "options": [
                "A、通过RAG SDK实现大语言模型知识增强",
                "B、通过Index SDK实现搜索推荐应用",
                "C、通过Audio SDK实现语音交互应用",
                "D、通过Vision SDK实现图像智能分析"
            ],
            "correct_answer": ["A", "D"]
        },
        {"type": "multi_choice",
            "id": 1012,
            "question": "从AI4science的角度来看，科学是从微观粒子到宏观系统的连续体系，深度学习的方法在微观尺度上可以实现以下哪些功能?",
            "options": ["A、设计新的药物小分子", "B、设计蛋白质大分子", "C、设计高效的飞行器", "D、预测全球气象"],
            "correct_answer": ["A", "B"]
        },
        {"type": "multi_choice",
            "id": 1013,
            "question": "华为AI正在加速进入千行百业，以下哪些选项是华为AI可以赋能的行业?",
            "options": ["A、制造", "B、互联网", "C、运营商", "D、金融"],
            "correct_answer": ["A", "B", "C", "D"]
        },
        {"type": "multi_choice",
            "id": 1014,
            "question": "以下哪些选项是智算集群需要高速无损网络的原因?",
            "options": [
                "A、AI计算的迭代过程中，会有大量突发流量",
                "B、AI计算中如果有流发生延迟，会导致存储和计算资源无法充分利用",
                "C、AI计算中会有不间断的小流量产生",
                "D、AI计算发生丢包中断，只能从头开始训练"
            ],
            "correct_answer": ["A", "B"]
        },
        {"type": "multi_choice",
            "id": 1015,
            "question": "华为智能无损技术架构包含以下哪些层?",
            "options": ["A、应用加速层", "B、流量调度层", "C、拥塞控制层", "D、应用使能层"],
            "correct_answer": ["A", "B", "C"]
        }
    ];

 var    true_false_questions = [
        {"type": "true_false",
            "id": 2001,
            "question": "与CPU相比，NPU中ALU的占比更高。",
            "correct_answer": "A"
        },
        {"type": "true_false",
            "id": 2002,
            "question": "华为AI基础硬件包含昇腾芯片和GPU芯片。",
            "correct_answer": "B"
        },
        {"type": "true_false",
            "id": 2003,
            "question": "智算芯片适合逻辑简单、计算密集型的高并发任务。",
            "correct_answer": "A"
        },
        {"type": "true_false",
            "id": 2004,
            "question": "AscendCL(Ascend computing Language)是一套用于在CANN上开发深度神经网络推理应用的C语言API库，提供模型加载与执行、媒体数据处理、算子加载与执行等API。",
            "correct_answer": "A"
        },
        {"type": "true_false",
            "id": 2005,
            "question": "Mind Edge提供边缘AI业务基础组件管理和边缘AI业务容器的全生命周期管理能力，同时提供节点看管、日志采集等统一运维能力和严格的安全可信保障。",
            "correct_answer": "A"
        },
        {"type": "true_false",
            "id": 2006,
            "question": "在传统产业如制药、材料、育种、航空航天，新产品的研发往往需要五到十年的研发周期，AI4Science通过对暗默知识的学习和建模，可以在传统产业的设计阶段发挥巨大的价值，不仅可以加速新产品的研发，还可以提升产品设计的成功率，极大的提高生产力。",
            "correct_answer": "A"
        },
        {"type": "true_false",
            "id": 2007,
            "question": "当朋友发来一个账号信息，需要转账的时候，小艺可以自动完成转账操作。",
            "correct_answer": "B"
        },
        {"type": "true_false",
            "id": 2008,
            "question": "华为AI已助力金融行业实现内容生成，其中内容生成有智能摘要和报告写作两类模式，可以提升各类文档、报告撰写效率及质量。",
            "correct_answer": "A"
        },
        {"type": "true_false",
            "id": 2009,
            "question": "AskO3的智能助手中心允许工程师搭建个人专属智能助手。",
            "correct_answer": "A"
        },
        {"type": "true_false",
            "id": 2010,
            "question": "具身智能的核心是智能，生成式模型等AI技术的最新进展，实现了文本、视觉、语音等多种模态信息的理解和转换，将这些AI技术嵌入到物理实体机上，可以显著提升机器人对环境的感知、交互和任务执行能力。",
            "correct_answer": "A"
        }
    ];
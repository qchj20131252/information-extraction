# 信息提取基线系统-InfoExtractor
## 摘要
InfoExtractor是一个基于Schema约束知识提取数据集（SKED）的信息提取基线系统。 InfoExtractor采用具有p分类模型和so-labeling模型的流水线架构，这些模型都使用PaddlePaddle实现。 p分类模型是多标签分类，其使用具有最大池网络的堆叠Bi-LSTM来识别给定句子中涉及的谓词。 然后在这样的标记模型中采用BIEO标记方案的深Bi-LSTM-CRF网络，以标记主题和对象提及的元素，给出在p分类模型中区分的谓词。 InfoExtractor在开发集上的F1值为0.668。

## 开始
### 环境要求
Paddlepaddle v1.2.0 <br />
Numpy <br />
内存要求10G用于训练，6G用于推断

### Step 1: 安装paddlepaddle
目前我们只在PaddlePaddle Fluid v1.2.0上进行了测试，请先安装PaddlePaddle，然后在[PaddlePaddle主页]((http://www.paddlepaddle.org/))上查看有关PaddlePaddle的更多详细信息。

### Step 2: 下载训练数据，开发数据和schema文件
请从[竞赛网站](http://lic2019.ccf.org.cn/kg)下载训练数据，开发数据和架构文件，然后解压缩文件并将它们放在./data/文件夹中。
```
cd data
unzip train_data.json.zip 
unzip dev_data.json.zip
cd -
```
### Step 3: 获取字典文件词典文件
从训练和开发数据的字段“text”中获取高频字，然后将这些高频词组成字典。
从训练和开发数据的字段“postag”中获取高频词，然后将这些高频词组成词典。
```
python lib/get_char.py ./data/train_data.json ./data/dev_data.json > ./dict/char_idx
python lib/get_vocab.py ./data/train_data.json ./data/dev_data.json > ./dict/word_idx
```
### Step 4: 训练p分类模型
首先，训练分类模型以识别句子中的谓词。 请注意，如果您需要更改默认的超参数，例如 隐藏层大小或是否使用GPU进行训练（默认情况下，使用CPU训练）等。请修改```/ conf / IE_extraction.conf```中的特定参数，然后运行以下命令：
```
python bin/p_classification/p_train.py --conf_path=./conf/IE_extraction.conf
```
经过训练的p分类模型将保存在文件夹```./ model / p_model```中。

### Step 5: 训练so-labeling模型
在获得句子中存在的谓词之后，训练序列标记模型以识别对应于出现在句子中的关系的s-o对。 <br />
在训练这样的标记模型之前，您需要准备符合训练模型格式的训练数据，以训练如此标记的模型。
```
python lib/get_spo_train.py  ./data/train_data.json > ./data/train_data.p
python lib/get_spo_train.py  ./data/dev_data.json > ./data/dev_data.p
```
要训​​练这样的标签模型，您可以运行：
```
python bin/so_labeling/spo_train.py --conf_path=./conf/IE_extraction.conf
```
经过训练的so-labeling模型将保存在文件夹```./ model / spo_model```中。

### Step 6: 用两个经过训练的模型进行推断
训练结束后，您可以选择经过训练的预测模型。以下命令用于使用最后一个模型进行预测。您还可以使用开发集来选择最佳预测模型。要使用带有演示测试数据的两个训练模型进行推理（在```/。/ data / test_demo.json```下），请分两步执行命令：
```
python bin/p_classification/p_infer.py --conf_path=./conf/IE_extraction.conf --model_path=./model/p_model/final/ --predict_file=./data/test_demo.json > ./data/test_demo.p
python bin/so_labeling/spo_infer.py --conf_path=./conf/IE_extraction.conf --model_path=./model/spo_model/final/ --predict_file=./data/test_demo.p > ./data/test_demo.res
```
预测的SPO三元组将保存在文件夹```./ data / test_demo.res```中。

## 评估
精度、召回率和F1分数是衡量参与系统性能的基本评价指标。在获得模型的预测三元组之后，可以运行以下命令。<br />
考虑到数据安全性，我们不提供别名字典。
```
zip -r ./data/test_demo.res.zip ./data/test_demo.res
python bin/evaluation/calc_pr.py --golden_file=./data/test_demo_spo.json --predict_file=./data/test_demo.res.zip
```

## 讨论
如果您有任何问题，可以在github上提交一个问题，我们会定期回复您。 </br>

##版权和许可
版权所有2019 Baidu.com，Inc。保留所有权利 <br />
根据Apache许可证2.0版（“许可证”）获得许可; 除非符合许可，否则您不得使用此文件。 您可以在此处获得许可副本 <br />
http://www.apache.org/licenses/LICENSE-2.0 <br />
除非适用法律要求或书面同意，否则根据许可证分发的软件将按“原样”分发，不附带任何明示或暗示的担保或条件。 有关管理许可下的权限和限制的特定语言，请参阅许可证。

##附录
在发布的数据集中，句子的字段postag表示句子的分割和词性标注信息。词性标注(PosTag)的缩略语及其对应的词性意义见下表。<br />
此外，数据集的给定分段和词性标注仅是参考，可以用其他分段结果替换。<br />

##词性标记集
|Tag|Description|含义描述|Example|
|:---|:---|:---|:---|
|r|pronoun|代词|我们|
|n|general noun|名词|苹果|
|ns|geographical name|地名|北京|
|wp|punctuation|标点|，。！|
|k|suffix|后缀|界, 率|
|h|prefix|前缀|阿, 伪|
|u|auxiliary|助词|的, 地|
|c|conjunction|连词|和, 虽然|
|v|verb|动词|跑, 学习|
|p|preposition|介词|在, 把|
|d|adverb|副词|很|
|q|quantity|量词|个|
|nh|person name|人名|杜甫, 汤姆|
|m|number|数词|一，第一|
|e|exclamation|语气词|哎|
|b|other noun-modifier|状态词|大型, 西式|
|a|adjective|形容词|美丽|
|nd|direction noun|方位词|右侧|
|nl|location noun|处所词|城郊|
|o|onomatopoeia|拟声词|哗啦|
|nt|temporal noun|时间词|近日, 明代|
|nz|other proper noun|其他专名|诺贝尔奖|
|nl|organization name|机构团体|保险公司|
|i|idiom|成语|百花齐放|
|j|abbreviation|缩写词|公检法|
|ws|foreign words|外来词	CPU|
|g|morpheme|词素|茨, 甥|
|x|non-lexeme|非词位|萄, 翱|

##依存句法分析标注关系
|关系类型|Tag|Description|Example|
|:---|:---|:---|:---|
|主谓关系|SBV|subject-verb|我送她一束花(我 \<-- 送)|
|动宾关系|VOB|直接宾语，verb-object|我送她一束花 (送 --> 花)|
|间宾关系|IOB|间接宾语，indirect-object|我送她一束花 (送 --> 她)|
|前置宾语|FOB|前置宾语，fronting-object|他什么书都读 (书 \<-- 读)|
|兼语|DBL|double|他请我吃饭 (请 --> 我)|
|定中关系|ATT|attribute|红苹果 (红 \<-- 苹果)|
|状中结构|ADV|adverbial|非常美丽 (非常 \<-- 美丽)|
|动补结构|CMP|complement|做完了作业 (做 --> 完)|
|并列关系|COO|coordinate|大山和大海 (大山 --> 大海)|
|介宾关系|POB|preposition-object|在贸易区内 (在 --> 内)|
|左附加关系|LAD|left adjunct|大山和大海 (和 \<-- 大海)|
|右附加关系|RAD|right adjunct|孩子们 (孩子 --> 们)|
|独立结构|IS|independent structure|两个单句在结构上彼此独立|
|标点|WP|punctuation|。|
|核心关系|HED|head|指整个句子的核心|

##命名实体说明
|标记|说明|
|:---|:---|
|Nh|人名|
|Ns|地名|
|Ni|机构名|
前缀说明: 包含BIES四种前缀，分别表示 开始、中间、结束、独立

##语义角色列表
|标记|说明|
|:---|:---|
|ADV|adverbial, default tag ( 附加的，默认标记 )|
|BNE|beneﬁciary ( 受益人 )|
|CND|condition ( 条件 )|
|DIR|direction ( 方向 )|
|DGR|degree ( 程度 )|
|EXT|extent ( 扩展 )|
|FRQ|frequency ( 频率 )|
|LOC|locative ( 地点 )|
|MNR|manner ( 方式 )|
|PRP|purpose or reason ( 目的或原因 )|
|TMP|temporal ( 时间 )|
|TPC|topic ( 主题 )|
|CRD|coordinated arguments ( 并列参数 )|
|PRD|predicate ( 谓语动词 )|
|PSR|possessor ( 持有者 )|
|PSE|possessee ( 被持有 )|
备注: 核心的语义角色为A0-5六种，A0通常表示动作的施事，A1通常表示动作的影响等，A2-5根据谓语动词不同会有不同的语义含义。

##语义依存关系说明
|关系类型|Tag|Description|Example|
|:---|:---|:---|:---|
|施事关系|Agt|Agent|我送她一束花 (我 \<-- 送)|
|当事关系|Exp|Experiencer|我跑得快 (跑 --> 我)|
|感事关系|Aft|Affection|我思念家乡 (思念 --> 我)|
|领事关系|Poss|Possessor|他有一本好读 (他 \<-- 有)|
|受事关系|Pat|Patient|他打了小明 (打 --> 小明)|
|客事关系|Cont|Content|他听到鞭炮声 (听 --> 鞭炮声)|
|成事关系|Prod|Product|他写了本小说 (写 --> 小说)|
|源事关系|Orig|Origin|我军缴获敌人四辆坦克 (缴获 --> 坦克)|
|涉事关系|Datv|Dative|他告诉我个秘密 ( 告诉 --> 我 )|
|比较角色|Comp|Comitative|他成绩比我好 (他 --> 我)|
|属事角色|Belg|Belongings|老赵有俩女儿 (老赵 \<-- 有)|
|类事角色|Clas|Classification|他是中学生 (是 --> 中学生)|
|依据角色|Accd|According|本庭依法宣判 (依法 \<-- 宣判)|
|缘故角色|Reas|Reason|他在愁女儿婚事 (愁 --> 婚事)|
|意图角色|Int|Intention|为了金牌他拼命努力 (金牌 \<-- 努力)|
|结局角色|Cons|Consequence|他跑了满头大汗 (跑 --> 满头大汗)|
|方式角色|Mann|Manner|球慢慢滚进空门 (慢慢 \<-- 滚)|
|工具角色|Tool|Tool|她用砂锅熬粥 (砂锅 \<-- 熬粥)|
|材料角色|Malt|Material|她用小米熬粥 (小米 \<-- 熬粥)|
|时间角色|Time|Time|唐朝有个李白 (唐朝 \<-- 有)|
|空间角色|Loc|Location|这房子朝南 (朝 --> 南)|
|历程角色|Proc|Process|火车正在过长江大桥 (过 --> 大桥)|
|趋向角色|Dir|Direction|部队奔向南方 (奔 --> 南)|
|范围角色|Sco|Scope|产品应该比质量 (比 --> 质量)|
|数量角色|Quan|Quantity|一年有365天 (有 --> 天)|
|数量数组|Qp|Quantity-phrase|三本书 (三 --> 本)|
|频率角色|Freq|Frequency|他每天看书 (每天 \<-- 看)|
|顺序角色|Seq|Sequence|他跑第一 (跑 --> 第一)|
|描写角色|Desc(Feat)|Description|他长得胖 (长 --> 胖)|
|宿主角色|Host|Host|住房面积 (住房 \<-- 面积)|
|名字修饰角色|Nmod|Name-modifier|果戈里大街 (果戈里 \<-- 大街)|
|时间修饰角色|Tmod|Time-modifier|星期一上午 (星期一 \<-- 上午)|
|反角色|r + main role| |打篮球的小姑娘 (打篮球 \<-- 姑娘)|
|嵌套角色|d + main role| |爷爷看见孙子在跑 (看见 --> 跑)|
|并列关系|eCoo|event Coordination|我喜欢唱歌和跳舞 (唱歌 --> 跳舞)|
|选择关系|eSelt|event Selection|您是喝茶还是喝咖啡 (茶 --> 咖啡)|
|等同关系|eEqu|event Equivalent|他们三个人一起走 (他们 --> 三个人)|
|先行关系|ePrec|event Precedent|首先，先|
|顺承关系|eSucc|event Successor|随后，然后|
|递进关系|eProg|event Progression|况且，并且|
|转折关系|eAdvt|event adversative|却，然而|
|原因关系|eCau|event Cause|因为，既然|
|结果关系|eResu|event Result|因此，以致|
|推论关系|eInf|event Inference|才，则|
|条件关系|eCond|event Condition|只要，除非|
|假设关系|eSupp|event Supposition|如果，要是|
|让步关系|eConc|event Concession|纵使，哪怕|
|手段关系|eMetd|event Method|	|
|目的关系|ePurp|event Purpose|为了，以便|
|割舍关系|eAban|event Abandonment|与其，也不|
|选取关系|ePref|event Preference|不如，宁愿|
|总括关系|eSum|event Summary|总而言之|
|分叙关系|eRect|event Recount|例如，比方说|
|连词标记|mConj|Recount Marker|和，或|
|的字标记|mAux|Auxiliary|的，地，得|
|介词标记|mPrep|Preposition|把，被|
|语气标记|mTone|Tone|吗，呢|
|时间标记|mTime|Time|才，曾经|
|范围标记|mRang|Range|都，到处|
|程度标记|mDegr|Degree|很，稍微|
|频率标记|mFreq|Frequency Marker|再，常常|
|趋向标记|mDir|Direction Marker|上去，下来|
|插入语标记|mPars|Parenthesis Marker|总的来说，众所周知|
|否定标记|mNeg|Negation Marker|不，没，未|
|情态标记|mMod|Modal Marker|幸亏，会，能|
|标点标记|mPunc|Punctuation Marker|，。！|
|重复标记|mPept|Repetition Marker|走啊走 (走 --> 走)|
|多数标记|mMaj|Majority Marker|们，等|
|实词虚化标记|mVain|Vain Marker| |
|离合标记|mSepa|Seperation Marker|吃了个饭 (吃 --> 饭) 洗了个澡 (洗 --> 澡)|
|根节点|Root|Root|全句核心节点|
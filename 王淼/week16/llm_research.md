# 开源大模型架构调研报告
## 作业说明
本次调研另外选取4个主流开源权重模型：Llama‑4 Scout、Mistral‑3 Large、GPT‑OSS‑120B、Jamba‑2，从注意力机制、MoE/FFN、位置编码、归一化稳定化、上下文能力、核心创新点开展调研，并与课程所学模型做路线对比。

## 1 Llama‑4 Scout（Meta）
- 总参/激活：109B总参 /17B每token激活，MoE架构
- 上下文：原生支持10M token（千万级上下文）
- 许可证：Llama Community License（非OSI开源，月活7亿以上需商业签约）

### 架构细节
1. **注意力模块：GQA分组查询注意力**
未使用MLA，沿用GQA（每8个Q头共享一组KV）。采用**iRoPE（interleaved RoPE‑NoPE）交替层设计**：部分层启用RoPE，部分层完全NoPE。
- RoPE层负责局部近距离语义；NoPE层负责超长距离全局依赖。
- NoPE层增加attention temperature缩放，缓解超长序列softmax分数衰减。

2. **MoE‑FFN：少而大专家路线**
共16个专家，每token激活 top‑1+1（1路由专家 +1共享专家）。
> 和DeepSeek“大量小专家”路线相反：专家数量少、单个专家隐维很大；路由器简单，训练稳定性好，但细分任务表达能力偏弱。
- 使用辅助损失实现负载均衡，没有采用DeepSeek的noaux无辅助损失方案。
- 全部层均为MoE层，无Dense稠密层穿插。
- FFN激活函数：SwiGLU。

3. **归一化与稳定化**
- 基础归一化：RMSNorm；RoPE层对Q/K做额外RMSNorm，抑制logits爆炸。
- 残差：标准单路残差连接，无mHC、AttnRes等多路残差改造。

4. **多模态**
原生早期融合多模态，文本、图像从第一层联合处理，不是外挂ViT‑projector模式。

### 优缺点
✅优点：千万级超长上下文，微调工具链生态成熟，MoE训练稳定。
❌缺点：无MLA优化，KV‑Cache显存开销高于DS‑V4/Kimi‑K3；依赖辅助损失；商业使用存在许可门槛。

> 路线对照课程PPT：DS‑V4采用CSA+HCA块压缩实现长上下文；Llama‑4 Scout依靠iRoPE交替RoPE‑NoPE + GQA实现千万上下文。

## 2 Mistral‑3 Large（Mistral AI）
- 总参/激活：675B总参 /41B激活参数，MoE架构
- 上下文：原生256K，支持YaRN外推至1M token
- 许可证：非完全开源，可下载非商用权重

### 架构细节
1. **注意力模块：MLA多头隐变量注意力**
跟进DeepSeek V2/V3的MLA设计，缓存低维latent向量代替完整K/V，降低KV‑Cache显存占用。
- 不使用DSA、CSA、线性注意力；仍然是完整注意力内核，复杂度$O(L^2)$。
- 位置编码：标准RoPE，搭配YaRN做长度外推。

2. **MoE‑FFN**
共64专家，top‑8激活+1共享专家；SwiGLU激活；借鉴DeepSeek‑V3，采用noaux无辅助损失负载均衡。
- MoE层与Dense稠密层交替排布，一半层MoE、一半层Dense，兼顾通用能力与稀疏算力。

3. **归一化与稳定化**
RMSNorm + QK‑Norm；标准单路残差；无mHC / AttnRes；不做激活钳制，依靠BF16/FP8混合精度保证数值稳定。

4. **多模态**
原生ViT多模态，通过projector将视觉特征对齐文本隐空间。

### 优缺点
✅优点：MLA降低KV显存；noaux均衡无辅助损失；MoE/Dense交替设计通用性好。
❌缺点：未消除平方复杂度，1M上下文训练推理算力开销大，没有稀疏/线性注意力优化。

> 路线对照课程PPT：仅采用MLA，没有引入稀疏、线性、块压缩，属于MLA基础改良派。

## 3 GPT‑OSS‑120B（OpenAI）
- 总参/激活：117B总参，每token激活51B
- 上下文：128K
- 许可证：Apache‑2.0，可商用开源

### 架构细节
1. **注意力**：标准GQA，RoPE位置编码；无MLA、无线性/稀疏注意力。
2. **MoE‑FFN**：32专家，top‑8激活；SwiGLU；借鉴DS‑V4做硬钳制激活，输出限制`[-10, 10]`，适配MXFP4量化感知训练QAT。
3. **归一化**：RMSNorm + QK‑Norm；标准单路残差，无mHC、AttnRes。
4. **推理特性**：原生支持思考模式输出`reasoning_content`，支持`reasoning_effort`配置 low/high/max。
5. **优化器**：AdamW，未使用Muon。

### 优缺点
✅优点：Apache2.0完全可商用；原生思维链输出；硬钳制激活提升低精度量化稳定性。
❌缺点：缺少长上下文专项优化；无MLA/稀疏，长文本KV‑Cache开销高。

> 路线对照课程PPT：吸收DS‑V4有界激活钳制的经验，但没有采纳Muon、mHC、CSA/HCA等复杂架构改动，偏向稳妥工程路线。

## 4 Jamba‑2（AI21 Labs）
> 非纯Transformer架构，**Transformer + Mamba‑2 SSM状态空间混合架构**；Apache‑2.0开源；上下文256K，可外推1M+。

### 架构细节
1. **层排布：Mamba‑2 SSM层与Transformer层交替**
- SSM(Mamba‑2)：无QKV注意力，状态空间模型，计算复杂度$O(L)$；维护内部状态，不需要KV‑Cache，擅长长距离依赖。
- 每隔若干层SSM插入一层GQA Transformer注意力层，用来兜底精确原文检索。设计思想类似Kimi‑K3“多层线性注意力+少量全量注意力兜底”。
2. 位置编码：SSM自带位置感知；Transformer层使用RoPE。
3. FFN：Dense稠密SwiGLU，**无MoE**，全稠密模型。
4. 归一化：RMSNorm；标准残差连接。

### 优缺点
✅优点：SSM带来线性复杂度，长上下文显存友好；无MoE，部署简单。
❌缺点：全Dense架构，模型扩容后每token计算量同步上升；精确原文检索能力依赖少量Transformer层兜底。

> 路线对照课程PPT：Kimi‑K3/Qwen3.6是Transformer内部改造（线性注意力+全量层）；Jamba‑2属于跨范式混合：SSM状态空间 + Transformer注意力模块。

## 5 调研模型横向对比
|模型|Llama‑4 Scout|Mistral‑3 Large|GPT‑OSS‑120B|Jamba‑2|
|---|---|---|---|---|
|总参/激活|109B /17B|675B /41B|117B /51B|Dense，无MoE|
|注意力方案|GQA + iRoPE(RoPE‑NoPE交替)|MLA|GQA|Mamba2‑SSM +间隔GQA|
|长上下文手段|iRoPE交替层，原生10M|MLA+YaRN外推|RoPE外推|SSM状态空间O(L)|
|MoE专家配置|16专家 top‑1+1，辅助损失|64专家 top‑8+1 noaux|32专家 top‑8|无MoE(Dense)|
|激活处理|SwiGLU无钳制|SwiGLU无钳制|SwiGLU hard clamp [-10,10]|SwiGLU无钳制|
|残差改造|标准残差|标准残差|标准残差|标准残差|
|优化器|AdamW|AdamW|AdamW|AdamW|
|许可证|Llama社区协议|非商用权重|Apache‑2.0|Apache‑2.0|

## 6 总结（结合课程PPT）
课程学习的DS‑V4、GLM‑5.2、Kimi‑K3、Qwen3.6全部基于纯Transformer解码器内部做改造：分别采用块压缩CSA/HCA、稀疏DSA、线性注意力、MLA。本次调研4个开源模型展示另外几条技术路线：

1. **Llama‑4 Scout**：GQA基础上依靠iRoPE交替RoPE‑NoPE实现千万上下文；MoE走少而大专家+辅助损失，与DeepSeek多小专家+noaux路线形成鲜明对比。
2. **Mistral‑3 Large**：只采纳MLA做KV压缩，不引入稀疏、压缩、线性注意力，保留$O(L^2)$注意力内核。
3. **GPT‑OSS‑120B**：借鉴DS‑V4有界钳制激活，适配低精度QAT；其余架构改动克制，追求工程稳妥。
4. **Jamba‑2**：跳出纯Transformer，引入Mamba‑2 SSM状态空间，模块级混合拿到线性复杂度。

### 行业共同趋势
1. 原始MHA基本淘汰，GQA、MLA成为MoE大模型标配。
2. 实现超长上下文有多条可行路径：块压缩、稀疏选择、线性注意力、iRoPE交替、SSM状态空间，不存在万能银弹。
3. MoE分化两大流派：大量细粒度小专家；少量大专家，各有利弊。

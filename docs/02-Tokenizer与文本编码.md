# 02. Tokenizer 与文本编码：模型如何理解提示词

## 1. 三个容易混淆的角色

| 角色 | 输入 | 输出 | 负责 |
|---|---|---|---|
| Tokenizer | 字符串 | token id 序列 | 切分文本 |
| Embedding | token id | 向量 | 把编号变成可学习表示 |
| Text Encoder | token id 或 embedding | 语义向量序列 | 结合上下文提取含义 |

一句话总结：

```text
tokenizer 负责切，embedding 负责查表，text encoder 负责理解上下文。
```

## 2. Tokenizer 不是“按字切”

以英文 BPE 类 tokenizer 为例，常见单词可能对应一个 token，罕见词会被拆成多个子词：

```text
astronaut
→ 可能是一个 token

unbelievable
→ 可能拆成 un + believe + able
```

这样做的原因：

- 词汇表不能无限大；
- 罕见词、新词、错拼词仍能被表示；
- 相同子词可以复用；
- 训练和推理必须使用完全一致的词表和切分规则。

## 3. BPE：`vocab.json` + `merges.txt`

本仓库中的 Stable Diffusion 和 Qwen 都使用 BPE/BPE 类 tokenizer。

BPE 的基本流程：

1. 先把文本拆成很小的基础单元；
2. 统计训练语料中哪些相邻单元经常一起出现；
3. 把高频组合合并成新单元；
4. 重复合并，直到达到设定词表大小。

因此你会看到两个关键文件：

| 文件 | 作用 |
|---|---|
| `vocab.json` | 子词 → id 的字典 |
| `merges.txt` | BPE 合并规则，决定优先怎么拼 |

例如 `merges.txt` 中可能出现：

```text
a b
i n
a b o u t
```

这些规则决定了 tokenizer 的切分行为。

## 4. SD1.5 的 CLIP tokenizer

查看本地目录：

```text
models/stable-diffusion-v1-5/tokenizer/
├── merges.txt
├── special_tokens_map.json
├── tokenizer_config.json
└── vocab.json
```

`vocab.json` 约 1.0 MB，`merges.txt` 约 0.5 MB。

CLIP 文本编码器配置中有一个很关键的数字：

```json
"max_position_embeddings": 77
```

这表示 CLIP 的文本路径通常最多处理 77 个 token。超过长度会被截断。因此 SD1.5 的 prompt 不是越长越好。

## 5. Qwen 的 tokenizer

Qwen 使用更大的 Qwen tokenizer：

```text
models/Qwen-Image-2512/tokenizer/
├── added_tokens.json
├── chat_template.jinja
├── merges.txt
├── special_tokens_map.json
├── tokenizer_config.json
└── vocab.json
```

它的文本编码器配置显示：

```json
"vocab_size": 152064,
"hidden_size": 3584,
"max_position_embeddings": 128000
```

与 CLIP 相比：

| 项目 | SD1.5 CLIP | Qwen2.5-VL |
|---|---:|---:|
| 词表规模 | 49,408 | 152,064 |
| hidden size | 768 | 3,584 |
| 最大位置长度 | 77 | 128,000 |
| 网络层数 | 12 | 28 |

这说明 Qwen 的文本理解能力远强于 SD1.5 的 CLIP，但代价是文本编码器本身就非常大。

## 6. token id 到 embedding

embedding 可以理解成一张巨大的查表：

```python
# 概念示例
embedding = torch.nn.Embedding(vocab_size, hidden_size)
x = embedding(token_ids)
```

如果：

```text
vocab_size = 152064
hidden_size = 3584
dtype = bfloat16
```

那么这张表的权重大小约为：

```text
152064 × 3584 × 2 bytes ≈ 1.04 GiB
```

所以不要以为 tokenizer 只是小文件；真正的 embedding 权重通常在模型权重里，可能很大。

## 7. Text Encoder 做了什么？

embedding 只表示“这个词本身是什么”。text encoder 还要让词的表示受到上下文影响。

例如：

```text
bank river
bank money
```

同一个 `bank` 在不同上下文中应有不同含义。Transformer text encoder 通过 attention 更新每个 token 的表示：

```text
token ids
  → embedding
  → attention + MLP + normalization
  → 上下文化语义向量
  → 交给图像去噪网络
```

## 8. 图像模型如何使用文本表示？

### Stable Diffusion v1.5

CLIP 输出文本向量后，UNet 通过 **cross-attention** 使用它们：

```text
UNet 的图像特征作为 Q
文本特征作为 K/V
```

直觉：图像位置去“询问”文本条件，决定这里应该画什么。

### Qwen-Image-2512

Qwen 的 transformer 配置：

```json
"joint_attention_dim": 3584,
"num_layers": 60
```

它使用更强的联合注意力机制，把文本 token 和图像 token 的信息放在同一个注意力空间中交互。理解成“图文一起开会”比“图像单向询问文本”更准确。

## 9. prompt 写法不是玄学，而是受编码器限制

不同模型的 prompt 理解方式不同：

- SD1.5 更依赖关键词、风格词、权重符号和社区经验；
- Qwen-Image 对自然语言长描述理解更好；
- 超过编码器最大长度会截断或需要特殊处理；
- 负面 prompt 的作用取决于管线是否启用 CFG。

## 10. 动手实验

激活环境后运行：

```python
from transformers import CLIPTokenizer

tok = CLIPTokenizer.from_pretrained(
    "models/stable-diffusion-v1-5/tokenizer", local_files_only=True
)

s = "a cat astronaut floating in space, cinematic lighting"
print(tok(s).input_ids)
print([tok.convert_ids_to_tokens(i) for i in tok(s).input_ids])
```

你会看到：

- 空格如何被编码；
- 标点如何变成 token；
- `</wof002>` 这类文件结束符如何附加；
- token id 与词汇表的对应关系。

## 11. 学习检查点

1. tokenizer 的输出为什么是整数？
2. `vocab.json` 和 `merges.txt` 各自有什么用？
3. embedding 与 text encoder 的分工是什么？
4. SD1.5 的 77 token 限制会带来什么问题？
5. 为什么 Qwen 的文本编码器比 SD1.5 大得多？

如果这些已经理解，请进入 [03-diffusers管线与模型加载.md](03-diffusers管线与模型加载.md)。

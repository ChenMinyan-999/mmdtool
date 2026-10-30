# mmdance — LLM 当编舞师，产出 MMD 可用的 VMD

把上一轮讨论的架构落成了一个可跑的 demo：

```
音乐 ──► L0 分析 ──► L1 编舞谱(JSON) ──► L2 动作合成 ──► L3 烘焙 ──► L4 校验
        (确定性)     (LLM / 规则)        (动作库+IK)     (VMD)      (指标)
```

**LLM 永远不写骨骼角度、不写帧号、不写骨骼名。** 它只写"第几拍用什么动作"的编舞谱；
角度、四元数、30fps 量化、脚部锁定由确定性代码完成。这就是这套方案能稳定落地的那条分界线。

---

## 1. 快速开始（当前机器零安装即可跑）

```powershell
cd D:\huawei\open\project

# 完整跑一遍：生成示例音乐 → 分析 → 编舞 → 合成 → VMD → 校验 → 预览
py -m mmdance.cli demo

# 只跑管线不渲染（1.5 秒出结果）
py -m mmdance.cli demo --no-render

# 换自己的音乐（mp3/flac 需要 ffmpeg 在 PATH 里，本机已有）
py -m mmdance.cli build --wav D:\music\song.mp3 --out out_song

# 只出编舞谱，人工改完再回灌
py -m mmdance.cli plan  --wav song.mp3 --out out_song/score.json
py -m mmdance.cli build --wav song.mp3 --score out_song/score.json --out out_song

# 或者一条命令（自动挑对解释器）
.\run_demo.ps1
```

> **解释器注意**：本机 `python` 指向 3.8.6，跑不了（需要 3.10+）；请用 `py`（=3.14.8）。
> `run_demo.ps1` 会自动按 `py -3.14` → `py -3` → `python` 的顺序探测可用解释器。

依赖：Python 3.10+、`numpy`、`Pillow`；`ffmpeg` 可选（编码 mp4 / 解码非 wav）。
本机实测环境是 Python 3.14.8 + numpy 2.5.3 + Pillow 12.3.0，都已具备，**无需安装任何东西**。

## 2. 产出物

| 文件 | 说明 |
|---|---|
| `out/motion.vmd` | **最终交付**。VMD 0002 格式，24 根骨骼 × 721 帧，MikuMikuDance 直接读 |
| `out/score.json` | 编舞谱。人可读可改，改完用 `--score` 重跑 |
| `out/music_analysis.json` | L0 的结构化音乐分析（BPM/拍点/段落/能量） |
| `out/report.json` | L4 校验报告 + VMD 结构检查 |
| `out/preview.mp4` / `preview.gif` | 预览动画（软件渲染，不需要 MMD） |
| `out/contact_sheet.png` | 9 个动作峰值帧拼图 |

示例曲目：`examples/demo_song.wav`，120 BPM、4/4、12 小节，由 `tools/make_demo_music.py` 合成，
无版权问题。

## 3. 五层各自做了什么

### L0 音乐分析（`mmdance/music.py`，确定性）

STFT → 对数幅度谱通量得到 onset 包络 → 自相关估计 BPM（带 120 BPM 先验抑制倍频错误）
→ 相位锁定节拍网格 → 按 t≈0 锚定小节线（绝大多数曲子/片段从小节线开始）→ 小节级 RMS 变点分段。

输出契约是 `MusicAnalysis`：`bpm / beat_times / beat_energy / sections`。想换 librosa、
beat_this 只替换这一个类，下游不动。

实测：120.045 BPM、49 拍、三段（intro/chorus/outro）、第 0 拍在 t=-3ms。

### L1 编舞谱（`mmdance/choreo.py`）

```json
{"beat": 8.0, "move": "arms_wide", "beats": 4, "weight": 0.88, "note": "chorus e=0.98"}
```

只有"拍号 + 动作名 + 时长"。三层保障：

1. **动作白名单**：`move` 必须命中 `mmdance/moves.py` 里的库，否则校验报错；
2. **结构校验**：`validate()` 检查重叠、越界、时长、能量匹配，错误直接拒绝，警告打印；
3. **可回退**：规则规划器（`plan()`）永远可用，LLM 失败时降级。

规则规划器的音乐决策（也是 LLM prompt 里的规则）：乐句 4 拍/2 拍；8 拍乐句头放"大动作"；
左右交替；段落能量决定词汇表。

### L2 动作合成（`mmdance/moves.py` + `synth.py`）

三层相加，再经运动学校正：

```
BASE_POSE（站立）  +  Σ 动作增量×权重  +  groove（弹跳/摆胯/甩臂，锁拍）
                            ↓
              FK → 地面钳制 → 脚部锁定 + 两骨 IK → 30fps 采样
```

- **动作库**：17 个用手写的语义姿势片段（`arm(side, up=150)` 这种，MPL 风格），
  每个片段是 `(phase, pose)` 关键帧，只覆盖它声明的通道，其余回落到 BASE_POSE。
- **融合**：按通道加权求和增量、权重和超过 1 时归一化，transition 由包络自动淡入淡出，不会跳变。
- **groove**：与节拍锁相的弹跳（用"下蹲深度 ↔ 膝角"的解析关系反算，保证脚不乱跑）、
  摆胯、躯干反向扭转、手臂摆动。
- **脚部锁定**：脚在接触状态时把踝关节水平位置钉住，用两骨 IK 反解膝角；抬脚自动释放。
  这是"脚滑"指标能压到 1.7 cm/s 的原因，也是看起来像在跳舞而不是飘的关键。
- **地面钳制**：摆胯会让一只脚沉到地面以下，直接整体抬升根骨骼（而不是掰腿）。

### L3 烘焙（`mmdance/vmd.py` + `synth.py:to_vmd`）

- 右手系 → MMD 左手系（Z 取反，四元数 z 分量取反）；
- 30fps 整数帧，逐帧关键帧（线性插值足矣，省掉贝塞尔控制点的坑）；
- 同时写 **FK 腿骨** 和 **足ＩＫ位置**：模型开 IK 时 MMD 解到我们给的目标点（位置一致，
  膝盖朝向可能因模型腿长略有差异），没开 IK 时 FK 也能还原同一姿势。这是双保险，
  避免"VMD 进 MMD 腿不动"的经典问题；
- 写完后**回读校验**（`validate.check_vmd`）：骨骼名白名单、关键帧不重复、四元数归一。

### L4 校验（`mmdance/validate.py`）

| 指标 | 含义 | 阈值 |
|---|---|---|
| `foot_slide` | 支撑脚每帧水平位移（p95，cm/s） | 3 / 8 |
| `ground_penetration` | 最低脚部低于地面的深度（cm） | 0.8 / 2.5 |
| `balance_outside` | 重心投影落在支撑多边形外的帧占比 | 22% / 45% |
| `joint_limit_hits` | 超关节限位的帧数 | 1 / 60 |
| `beat_alignment` | 身体最低点落在拍点 ±70ms 的比例 | ≥55% PASS |
| `snap` | 单关节角加速度 p90（检测抖动/抽帧感） | 6 / 10 |

当前示例曲目：全部 PASS，脚滑 1.7 cm/s、穿地 0.23 cm、拍点命中 97.9%（平均偏差 11ms）。

## 4. 接入真正的 LLM

LLM 只负责把"音乐结构 + 用户意图"翻译成谱面。调用点只有一个函数：
`choreo.plan_with_llm()`，OpenAI 兼容接口。

```powershell
$env:MMDANCE_API_KEY  = "sk-..."                       # 或 OPENAI_API_KEY
$env:MMDANCE_BASE_URL = "https://api.openai.com/v1"    # 可换任意兼容网关
py -m mmdance.cli build --wav song.mp3 --planner llm --out out_llm
```

细节：

- `build_llm_prompt()` 会把**音乐分析 JSON + 动作库清单（名称/时长/通道/能量/描述）**塞进 user prompt，
  system prompt 是硬规则（只能用库里的动作、不得重叠、段落能量匹配、8 拍乐句头放大动作……）；
- 返回必须是纯 JSON，解析后立刻 `validate()`；
- 校验失败会把错误清单塞回去重新生成（默认重试 2 次），仍失败则抛错，不会产出坏谱面；
- 建议：先跑 `plan` 出谱面 → 人眼扫一遍 → 用 `--score` 回灌。**人的 30 秒检查比 LLM 重试 5 次有用。**

换模型/换风格只需要动 `style` 和 prompt 里的词汇表。想扩动作库：往 `moves.py` 里加一个
`@move(...)` 装饰的函数，或用 `moves.export_catalog()` 导出 JSON 让 LLM 参照格式造新动作。

## 5. 在 MMD 里用

1. MMD 里加载一个 PMX 模型（需要含标准骨骼：センター/上半身/头/四肢；
   有 `上半身2` 更好，没有的话胸部扭转会丢失，其余照常）；
2. 菜单 → 文件 → 读取动作数据 → 选 `out/motion.vmd`；
3. 音频同步：把 mp3/wav 拖进 MMD 当"音频文件"，从第 0 帧开始播放即可
   （VMD 的帧号就是绝对时间 ×30，前后段本来就有空拍，对齐是准的）。

**可能出现的问题：**

| 现象 | 原因 | 处理 |
|---|---|---|
| 动作左右/前后镜像 | MMD 左手系约定差异 | `--no-flip` 重跑一次对比 |
| 腿完全不动 | 模型开了足 IK，但 VMD 没写 IK 目标 | 默认已写；若仍不动，改用 `--no-ik` 交给 FK |
| 膝盖朝反方向弯 | 目标点超出可达范围/模型腿长差异大 | 用 `--no-ik` 关掉 IK 目标，纯 FK |
| 脚陷进地面 | 模型身高与默认骨架(≈19.5 单位)差异大 | 改 `mmdance/rig.py` 的 rest 坐标按身高缩放 |
| 动作幅度偏大/偏小 | 模型骨骼比例差异 | 调 `--difficulty` |

> 说明：VMD 只存**局部旋转**，与模型骨骼长度无关，所以"动作幅度"天然可迁移；
> 只有需要绝对位置的 **センター平移** 和 **足ＩＫ目标** 依赖身高尺度。

## 6. 已知限制（诚实清单）

1. **动作是"库 + 程序化"，不是生成模型**。词汇量 17 个片段，能编出完整的、不重复的歌，
   但不会有 AIST++/diffusion 那种细腻的律动与个人风格。这是刻意的取舍：可控、可解释、零训练。
2. **节拍检测是启发式**。稳态电子/流行乐很准；自由速度、弱起、变速曲子会漂。
   可用 `--bpm` 手动指定；要真正鲁棒请换 `beat_this`。
3. **小节线锚定用"曲子从小节线开始"的先验**。如果输入是从半句中间切出来的片段，会差几拍。
4. **脚部锁定是启发式接触检测**，脚尖拖地/滑步这类风格化动作会被当成"接触"而钉住。
5. **没有碰撞检测**。手臂穿过头发/身体的情况没处理，只做了关节限位。
6. **不写 つま先ＩＫ**：脚趾 IK 段省略，模型的脚会保持平贴，平底舞步够用，脚趾细节动作会被限制。
7. **上身只有两根脊柱骨**（上半身 / 上半身2），做不出真正的 body wave。
8. **没做表情/口型 morph**，VMD 的 morph 段是空的（格式支持，`motion.add_morph()` 可用）。

## 7. 扩展路线（按性价比排序）

1. **动作库扩容 + 检索**（CustomDance 路线）：把片段做成带标签的库，LLM 只出"检索 query"，
   检索器按音乐相似度取 top-k 再拼接。收益最大、改动最小。
2. **接入 mixamo/BVH 重定向**：外部动作 → 标准骨架 → 现有烘焙链，动作质量立刻上一个档次。
3. **训练/接入 diffusion**（DanceChat 路线）：谱面当条件，生成器出动作，再走 L3/L4。
4. **闭环重规划**：把 `report.json` 的指标喂回 LLM（"第 3 段脚滑严重，换重心更稳的动作"）。

## 8. 开发与测试

```powershell
py -m unittest discover -s tests -v      # 30 个测试，约 1 秒
py tools\diagnose.py                     # 谱面 + 逐帧分层 + 最差帧定位
py tools\sheet.py                        # 只渲染动作峰值拼图（快速看姿态）
py tools\strip.py 240 8 4                # 渲染连续帧胶片条（看节奏与抖动）
py tools\shot.py 120 300 420             # 渲染指定帧
```

测试覆盖：四元数数学、MMD 骨骼名（含全角ＩＫ）与字节长度、VMD 读写往返与记录长度、
BPM/拍网格/分段、谱面校验（未知动作/重叠/越界必须报错）、动作库姿势有限性、
IK 唯一关键帧、四元数归一、无穿地、脚锁定漂移、报告结构、以及一次 4 秒的端到端 CLI。

## 9. 目录

```
mmdance/
  mathx.py     四元数/缓动/MMD 坐标转换
  rig.py       标准骨架、MMD 骨骼名映射、语义姿势助手、关节限位
  vmd.py       VMD 0002 读写（含 IK/相机/灯光段落的跳过）
  music.py     L0 音乐分析
  moves.py     动作库（17 个语义关键帧片段）
  choreo.py    L1 谱面模型、校验、规则规划器、LLM 适配器
  synth.py     L2 grove/融合/FK/地面钳制/脚锁/两骨 IK + L3 VMD 烘焙
  validate.py  L4 六项指标 + VMD 结构检查
  render.py    软件 3D 预览（透视、阴影、HUD）
  cli.py       命令行入口
tools/         生成示例音乐、诊断、拼图、胶片条
tests/         单元 + 集成测试
examples/      示例曲目
out/           产物
```

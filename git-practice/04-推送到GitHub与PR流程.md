# 笔记 04 · 把仓库推到 GitHub（含 PR 流程）

> 前置：网络已通（见 `02-nanogpt/notes/01-环境与网络实录.md` 第一节）
> 你的 GitHub 账号：`zhou512`，已有 3 个公开仓库

---

## 一、先搞清楚：本地仓库和远程仓库是两回事

```
D:\git-reproduce\          你的本地仓库（已经有 6 次提交）
        │
        │  git push
        ▼
github.com/zhou512/xxx     远程仓库（还不存在，需要先创建）
```

**`git push` 不会帮你创建远程仓库。** 这是新手第一个卡点。
所以顺序永远是：**先建远程仓库 → 再关联 → 再 push**。

---

## 二、创建远程仓库（网页操作，30 秒）

1. 打开 https://github.com/new
2. **Repository name** 填 `llm-from-scratch`（或你喜欢的名字）
3. **Description** 填：`从零复现 micrograd 与 nanoGPT，含中文语料实验与完整笔记`
4. 选择 **Public**（简历项目必须公开）
5. **★ 不要勾选** `Add a README file` / `.gitignore` / `license`
   —— 勾了会创建一个初始提交，导致你的本地仓库和远程仓库历史冲突
6. 点 **Create repository**

创建后 GitHub 会显示一堆命令，**不要照着它全部执行**（它会让你 `git init`，而你已经有仓库了）。

---

## 三、关联并推送（正确的做法）

```powershell
cd D:\git-reproduce

# 1. 关联远程仓库，起个名字叫 origin
git remote add origin git@github.com:zhou512/llm-from-scratch.git

# 2. 确认关联成功
git remote -v
# 应该看到：
# origin  git@github.com:zhou512/llm-from-scratch.git (fetch)
# origin  git@github.com:zhou512/llm-from-scratch.git (push)

# 3. 推送，-u 把 main 设成默认上游
git push -u origin main
```

之后每次提交完，只要 `git push` 就够了（因为 `-u` 记住了上游）。

### 用 SSH 还是 HTTPS？

| | SSH (`git@github.com:...`) | HTTPS (`https://github.com/...`) |
|---|---|---|
| 认证方式 | SSH 密钥（你已有 `id_ed25519`） | 凭据管理器 / Token |
| 是否要输密码 | 否 | 首次要，之后缓存 |
| 你的状态 | ✅ 实测 `Hi zhou512!` 通过 | ⚠️ 未验证凭据是否配好 |

**推荐 SSH**，因为你已经有密钥且实测可用。

> 想切协议：`git remote set-url origin https://github.com/zhou512/xxx.git`

---

## 四、常见错误对照

| 报错 | 原因 | 解决 |
|---|---|---|
| `remote origin already exists` | 已经关联过了 | `git remote set-url origin <新地址>` 或 `git remote remove origin` 重来 |
| `Repository not found` | 远程仓库不存在，或名字写错，或没权限 | 回第二步创建；检查大小写 |
| `failed to push some refs ... non-fast-forward` | 远程有本地没有的提交 | `git pull --rebase origin main` 再 push |
| `Permission denied (publickey)` | SSH 密钥没加到 GitHub | 把 `~/.ssh/id_ed25519.pub` 内容贴到 GitHub → Settings → SSH keys |
| `src refspec main does not match any` | 本地没有 main 分支 | `git branch -M main` 确认分支名 |
| 推送大文件被拒 | 超过了 100MB 限制 | 别提交数据集/权重（我们的 `.gitignore` 已经处理） |

---

## 五、PR（Pull Request）流程：为什么你需要它

**即使只有你一个人开发，也建议用 PR。** 原因：

1. **简历上能看到「会做 code review」**——这是团队协作的信号
2. PR 页面本身就是**项目变更的可读记录**
3. 养成「不在 main 上直接改」的习惯

### 完整流程

```powershell
# 1. 从最新的 main 开一个功能分支
git switch main
git pull                      # 确保是最新
git switch -c feat/micrograd-engine

# 2. 干活，提交（可以多次）
git add projects/01-micrograd/src/engine.py
git commit -m "feat: 实现 Value 类的自动求导"

# 3. 推分支到远程
git push -u origin feat/micrograd-engine

# 4. 网页上点 "Compare & pull request"
#    - 标题：feat: 实现 micrograd 自动求导引擎
#    - 描述：写清做了什么、为什么、怎么验证
#    - 创建 PR

# 5. 合并（自己 review 自己）
#    点 "Merge pull request" → "Confirm merge"

# 6. 回到本地，同步 main 并清理分支
git switch main
git pull
git branch -d feat/micrograd-engine
git push origin --delete feat/micrograd-engine
```

### PR 描述模板

```markdown
## 做了什么
实现 `Value` 类的反向传播，支持 + - * / ** relu 六种算子。

## 为什么
`loss.backward()` 在 PyTorch 里天天用，但梯度到底怎么算的不清楚。
这个 PR 从零实现链式法则，为后面读 nanoGPT 打基础。

## 怎么验证
```bash
cd projects/01-micrograd
.venv\Scripts\python.exe -m pytest tests/test_engine.py -v
```
23 个测试全部通过。

## 关键实现
- 用闭包 `_backward` 存每个算子的局部导数
- `backward()` 先做后序 DFS 得到拓扑序，再逆序遍历
- 梯度用 `+=` 累加（处理一个节点被多次使用的情况）
```

---

## 六、给这个项目的具体建议

**推荐做法**：把整个 `D:\git-reproduce` 作为**一个仓库**推上去，
而不是每个项目一个仓库。理由：

- micrograd 和 nanoGPT 共享笔记、工具、背景说明
- 单个仓库更容易展示「学习路径的连贯性」
- 简历上放一个仓库链接比放三个更容易讲

仓库名建议：`llm-from-scratch` 或 `llm-reproduction-notes`

**README 的写法**（面试官第一眼看的）：

```markdown
# 从零复现 LLM 核心组件

两个项目的完整复现记录，包含自己的实现、验收测试、学习笔记。

| 项目 | 内容 | 状态 |
|---|---|---|
| micrograd | 从零实现标量级自动求导引擎（111 行纯 Python） | ✅ |
| nanoGPT | 在单张 RTX 4070 8GB 上从零训练字符级 GPT | ✅ |
| 中文版 | 用 1.14M 字符的唐诗宋词语料重训 | ✅ |

## 亮点
- 自己写测试并用参考实现验证，抓到 4 处错误（见 commit 6417bbd）
- 中文语料的 uint16 溢出问题（nanoGPT 写死 dtype，词表 > 65535 会静默出错）
- 实测显存/训练速度，给出可复现的性能数据
```

---

## 七、你现在要做的

1. 决定仓库名
2. 在 GitHub 上创建空仓库（**不勾任何初始化选项**）
3. 跑这三条命令：

```powershell
cd D:\git-reproduce
git remote add origin git@github.com:zhou512/<仓库名>.git
git push -u origin main
```

4. 把结果发我，我帮你确认

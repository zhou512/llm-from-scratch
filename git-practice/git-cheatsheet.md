# Git 速查卡（为你定制）

> 定位：你已经会 `init / add . / push`。下面补的是**每天真正会用到、以及会救你命**的部分。
> 配套仓库：`D:\git-reproduce\git-practice\demo-repo`

---

## 0. 一次性的全局配置（你自己在终端跑，我的沙箱写不了 C:\Users\36339）

```bash
git config --global user.name "zhou512"
git config --global user.email "3633942545@qq.com"

# 默认分支叫 main，别再让 git 给你建 master
git config --global init.defaultBranch main

# Windows 换行符：让仓库里永远是 LF，检出到 Windows 时按需转换
git config --global core.autocrlf input

# 中文文件名不要在 status 里被显示成 \346\226\207 这种转义
git config --global core.quotepath false

# 以后 git pull 只允许快进合并，避免莫名其妙多出 merge commit
git config --global pull.ff only

# push 时只推当前分支到同名上游
git config --global push.default simple

# 凭据交给 Windows 凭据管理器，HTTPS 不用每次输密码
git config --global credential.helper manager
```

检查：

```bash
git config --global --list
```

---

## 1. 心智模型：三个区域

```
工作区 (Working Directory)   你正在编辑的文件
        │  git add
        ▼
暂存区 (Index / Staging)     你「挑选」出来的、准备提交的内容
        │  git commit
        ▼
版本库 (Repository / .git)   永久历史
```

**关键点**：`git add` 不是「保存」，是**挑选**。这正是 `git add .` 危险的原因——
它把所有改动一股脑扔进暂存区，包括你还没想好、甚至不该提交的东西。

### 看状态

```bash
git status            # 完整、人话版
git status -s         # 短格式：A=已暂存新增 M=修改 ??=未跟踪
git status --ignored  # 顺便列出被 .gitignore 忽略的文件
```

### 看差异（最容易搞混的一对）

```bash
git diff              # 工作区  vs 暂存区   （我还没 add 的改动）
git diff --cached     # 暂存区  vs  上次提交 （我这次要提交什么）★ 提交前必看
git diff HEAD         # 工作区+暂存区 vs 上次提交（全部改动）
```

### 提交

```bash
git add <file>        # 只加这一个文件 ← 推荐
git add -p            # ★ 交互式挑选「代码块」级别的改动，一个文件拆成多个提交
git commit -m "..."   # 提交
git commit --amend    # 改最近一次提交（还没 push 时用）
```

---

## 2. 提交信息规范（Conventional Commits）

格式：

```
<类型>: <一句话说清做了什么>

（可选正文：为什么这么做，而不是怎么做的）
```

常用类型：

| 类型 | 含义 | 例子 |
|---|---|---|
| `feat` | 新功能 | `feat: 添加 ResNet 训练脚本` |
| `fix` | 修 bug | `fix: 修正 dataloader 的 shuffle 未生效` |
| `docs` | 只改文档 | `docs: 补充环境安装说明` |
| `refactor` | 重构，行为不变 | `refactor: 抽出 evaluate() 函数` |
| `test` | 加/改测试 | `test: 补充边界条件用例` |
| `chore` | 杂活（依赖、配置） | `chore: 升级 torch 到 2.4` |

**原则**：一次提交 = 一个逻辑单元，且**能独立通过**。
反例：`git commit -m "update"`、`git commit -m "改了点东西"`。

> 为什么这对简历重要：面试官/HR 会点进你的 GitHub。
> 一串 `update` 的历史 vs 一串 `feat: xxx` 的历史，专业度差距一眼可见。

---

## 3. `.gitignore` 规则

```gitignore
__pycache__/      # 目录
*.log             # 通配
data/             # 数据集，通常几百 MB，绝不能进仓库
*.pth  *.ckpt     # 模型权重
.venv/            # 虚拟环境
.vscode/  .idea/  # 编辑器配置
```

注意：

- 已经被跟踪的文件，写进 `.gitignore` **不会**失效——要先 `git rm --cached <file>`
- 模板：https://github.com/github/gitignore  （Python.gitignore 直接抄）

---

## 4. 分支

```bash
git branch                     # 列出本地分支（* 是当前）
git switch -c feat/add-tests   # ★ 新建并切过去（新版写法）
git switch main                # 切回 main
git branch -m master main      # 重命名分支（我们这次干的就是这个）
git branch -d feat/add-tests   # 删除已合并的分支
git branch -D feat/xxx         # 强制删除（没合并也删，谨慎）
```

`git checkout -b` 是旧写法，和 `git switch -c` 等价。新项目建议用 `switch`。

### 分支命名约定

```
feat/xxx     新功能        fix/xxx      修 bug
docs/xxx     文档          exp/xxx      实验（复现项目里常用）
```

---

## 5. 合并

```bash
git switch main
git merge feat/add-tests                 # 默认：能快进就快进
git merge --no-ff feat/add-tests -m "..."# ★ 强制留一个 merge commit，历史更清楚
```

两种合并的区别：

- **Fast-forward**：main 没动过，直接把指针推过去，历史是一条直线，看不出「这里合并过一个分支」
- **--no-ff**：多一个 merge commit，PR 的边界清晰。做开源 PR 时 GitHub 一般帮你做这个

冲突了怎么办：

```bash
git status                 # 看哪些文件冲突
# 手动编辑文件，删掉 <<<<<<< ======= >>>>>>> 标记，改成你想要的最终内容
git add <冲突文件>
git merge --continue       # 或 git commit
# 想放弃这次合并：
git merge --abort
```

---

## 6. 撤销（重点，新手最需要）

### 6.1 `git restore` 到底动哪一层？

`restore` 默认只动**工作区**，这是最容易误会的地方。它**永远不碰历史**。

```bash
git restore <file>              # 工作区 ← 暂存区（没暂存就取 HEAD 版本）
git restore --staged <file>     # 暂存区 ← HEAD      （工作区不动！）
git restore --source=<hash> <file>  # 工作区 ← 指定的那次提交
```

记住这个方向：**箭头左边是被改写的，右边是数据来源。**

```
         git restore --staged        git restore
HEAD ────────────────► 暂存区 ────────────────► 工作区
（历史，restore 永远不碰这里）
```

三种典型场景（都在练习仓库里实测过）：

| 你想干什么 | 命令 | 后果 |
|---|---|---|
| 文件改乱了，想回到上次提交的样子 | `git restore gcd.py` | ⚠️ 工作区改动**永久丢失** |
| `git add` 错了，想撤出暂存区 | `git restore --staged gcd.py` | ✅ 文件内容**还在**，只是没被暂存 |
| 想拿某个旧版本的文件看看 | `git restore --source=1dd7c23 gcd.py` | ✅ 历史不变，只是工作区变了 |

**关键结论**：`git restore` 改的是**工作区/暂存区**，改不了**仓库**。
你手动把文件改好了，`git log` 里那条坏提交依然存在——修工作区 ≠ 修历史。

### 6.2 那怎么改「仓库」？

| 场景 | 命令 | 说明 |
|---|---|---|
| 提交信息写错 / 漏了文件（**未 push**） | `git commit --amend` | 改最近一次提交，哈希会变 |
| 撤销某次提交，改动**保留在暂存区** | `git reset --soft HEAD~1` | 只动 HEAD 指针 |
| 撤销某次提交，改动**退回工作区** | `git reset HEAD~1`（默认 `--mixed`） | 动指针 + 暂存区 |
| 撤销某次提交，改动**直接丢弃** | `git reset --hard HEAD~1` | ❌ 三样全动，危险 |
| 已经 push 了，想反做某次提交 | `git revert <hash>` | ✅ 产生一条**新提交**，历史不被改写 |

**记忆法**：`--soft` 只动指针，`--mixed`（默认）再动暂存区，`--hard` 三样全动。

**什么时候用 reset，什么时候用 revert？**
- 只有你本地有这些提交（没 push）→ `reset`，历史干净
- 已经 push 出去了 / 别人可能拉了 → `revert`，不要改写公共历史

### 6.3 `git reflog` 怎么读

`git reflog` 记录的是**HEAD 的每一次移动**，是本机的操作日志（`git log` 记的是「提交之间的关系」，两者完全不同）。

一行拆成四段：

```
1dd7c23 (HEAD -> main) HEAD@{0}: commit: feat:在gcd.py中加入了一行abc
   │                       │    │        └── 你当时用的提交信息
   │                       │    └── 发生了什么操作
   │                       └── 第几步之前（0 是最近）
   └── 那一步之后 HEAD 指向哪个提交
```

常见操作类型：

| 字样 | 含义 |
|---|---|
| `commit:` | 提交 |
| `commit (amend):` | `--amend` 改过的提交 |
| `checkout:` / `moving from X to Y` | 切分支 |
| `reset: moving to HEAD~1` | 被 reset 了 |
| `revert:` | 反做提交 |
| `merge ...` | 合并 |
| `Branch: renamed ...` | 重命名分支 |

### 6.4 用 reflog 救命

```bash
git reflog                       # 找到你要回到的那一步，比如 HEAD@{3}
git show HEAD@{3}                # 先确认内容对不对，别急着动
git switch -c rescue HEAD@{3}    # ★ 推荐：新建分支去接住它，最安全
# 或者直接挪回去（会改写当前分支）：
git reset --hard HEAD@{3}
```

`git reflog <branch>` 看某个分支的 reflog，`git reflog --date=iso` 带时间。

**为什么能救回来**：Git 里的提交对象在被垃圾回收（默认 90 天后）之前一直存在，
`reset` 只是把分支指针挪开了，对象还在。`reflog` 就是通往这些"孤儿提交"的索引。

> 注意：`reflog` 是**本机的**，不会 push 到 GitHub。换台电脑就没了。
> 所以别把 reflog 当备份策略。

---

## 7. 远程 & PR 流程

```bash
git remote add origin git@github.com:zhou512/demo-repo.git
git remote -v                       # 检查
git push -u origin main             # 第一次推，-u 记住上游
git push                            # 之后直接推
```

### 做一次 PR（开源贡献的标准动作）

```bash
# 1. 先 fork（网页上点 Fork），然后克隆你自己的 fork
git clone git@github.com:zhou512/some-project.git
cd some-project

# 2. 加上游仓库，方便同步别人的更新
git remote add upstream https://github.com/原作者/some-project.git

# 3. 从最新的 main 开分支
git switch main
git pull upstream main
git switch -c fix/typo-in-readme

# 4. 改代码，提交
git add <file>
git commit -m "fix: 修正 README 中的安装命令拼写"

# 5. 推到自己的 fork
git push -u origin fix/typo-in-readme

# 6. 网页上点 "Compare & pull request"，写清 what / why / how to test
```

PR 描述模板：

```markdown
## 做了什么
修正 README 中 pip install 命令的包名拼写错误。

## 为什么
按原命令安装会报 ModuleNotFoundError。

## 怎么验证
在干净虚拟环境里执行修正后的命令，安装成功。
```

---

## 8. 复现开源项目时的推荐 git 用法

这是你最关心的场景：

```bash
# 1. 克隆别人的项目
git clone <url>
cd <project>

# 2. ★ 先记录你拿到的原始状态
git switch -c baseline          # 留一条干净分支做对照
git switch -c exp/my-first-run  # 所有实验都在自己分支上做

# 3. 每跑通一步就提交一次（哪怕只是能跑起来）
git add -A && git commit -m "chore: 跑通官方 demo，记录环境"

# 4. 改代码前先看清楚改了什么
git diff

# 5. 想试一个可能失败的想法 → 开新分支，失败了直接删掉
git switch -c exp/try-attention
# ... 失败 ...
git switch main
git branch -D exp/try-attention   # 干净利落，main 完全没被污染
```

**核心原则：`main` 永远是可运行、可展示的状态。**
所有脏活、实验、失败尝试，都发生在分支上。这样你的 GitHub 主页任何时候点进去都是干净的。

---

## 9. 每天会用的五条命令

```bash
git status -s          # 我现在在哪、有什么改动
git diff --cached      # 我这次要提交什么（提交前必看）
git add -p             # 挑着暂存
git commit -m "feat: ..."
git log --oneline --graph --decorate   # 我做过什么
```

# 用「本地模拟远程仓库」完整演练 PR 流程
#
# 为什么需要这个：
# 真正的 PR 流程需要 GitHub 账号 + 网络 + 在网页上点按钮。
# 但 git 的「分支 -> 推送 -> 合并 -> 清理」这套动作完全可以本地演练，
# 用一个「裸仓库」当远程就行。等你在 GitHub 上操作时，动作是一模一样的。
#
# 用法（在任意目录）：
#   pwsh -File git-practice\local-remote\demo-pr.ps1
#
# 跑完你会看到每一步的命令和输出。

$ErrorActionPreference = "Stop"
$root = Join-Path $PSScriptRoot "workdir"
if (Test-Path $root) { Remove-Item -Recurse -Force $root }
New-Item -ItemType Directory -Force -Path $root | Out-Null

function Step($n, $msg) {
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor DarkGray
    Write-Host "步骤 $n : $msg" -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor DarkGray
}

function Run($cmd) {
    Write-Host "`$ $cmd" -ForegroundColor Yellow
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne $null -and $LASTEXITCODE -ne 0) {
        # git 有些命令正常退出码不是 0（比如 git diff --quiet），这里不强制中断
    }
}

# ---------------------------------------------------------------- 1. 造一个"远程仓库"

Step 1 "创建一个裸仓库当「远程」（模拟 github.com/you/repo.git）"
$remote = Join-Path $root "remote.git"
Run "git init --bare --initial-branch=main `"$remote`""
Write-Host "远程仓库路径: $remote" -ForegroundColor Green

# ---------------------------------------------------------------- 2. 模拟"初始项目"

Step 2 "造一个初始项目并推到远程（模拟别人已有的仓库）"
$seed = Join-Path $root "seed"
New-Item -ItemType Directory -Force -Path $seed | Out-Null
Push-Location $seed
Run "git init --initial-branch=main"
Run "git config user.name 'zhou512'"
Run "git config user.email '3633942545@qq.com'"
Set-Content -Path "README.md" -Value "# demo`n`n初始项目。" -Encoding UTF8
Run "git add README.md"
Run "git commit -m 'docs: 初始提交'"
Run "git remote add origin `"$remote`""
Run "git push -u origin main"
Pop-Location

# ---------------------------------------------------------------- 3. 克隆（模拟 fork 后 clone）

Step 3 "克隆「远程」到工作目录（模拟你 clone 自己的 fork）"
$work = Join-Path $root "work"
Run "git clone `"$remote`" `"$work`""
Push-Location $work
Run "git config user.name 'zhou512'"
Run "git config user.email '3633942545@qq.com'"

# ---------------------------------------------------------------- 4. 开分支干活

Step 4 "★ 开一个功能分支（绝不在 main 上直接改）"
Run "git switch -c feat/add-gcd"
$gcdContent = @'
def gcd(a, b):
    while b:
        a, b = b, a % b
    return abs(a)
'@
Set-Content -Path "gcd.py" -Value $gcdContent -Encoding UTF8
Run "git status -s"
Run "git add gcd.py"
Run "git commit -m 'feat: 添加 gcd 函数'"
Write-Host ""
Write-Host "现在 main 上有 gcd.py 吗？" -ForegroundColor Magenta
Run "git switch main"
Write-Host "main 上文件列表:"
Get-ChildItem -File | Select-Object -ExpandProperty Name
Run "git switch feat/add-gcd"

# ---------------------------------------------------------------- 5. 推送分支

Step 5 "把分支推到远程（这一步之后才能在网页上开 PR）"
Run "git push -u origin feat/add-gcd"

# ---------------------------------------------------------------- 6. 模拟"开 PR"

Step 6 "模拟 PR：远程会显示「feat/add-gcd 想合进 main」"
Run "git log --oneline origin/main..feat/add-gcd"
Write-Host "↑ 这些就是 PR 里会展示的提交" -ForegroundColor Green

# ---------------------------------------------------------------- 7. 合并

Step 7 "★ 合并 PR（--no-ff 保留合并痕迹，和 GitHub 点 Merge 效果一致）"
Run "git switch main"
Run "git merge --no-ff feat/add-gcd -m 'merge: 合入 gcd 函数 (PR #1)'"
Run "git log --oneline --graph --decorate"
Run "git push origin main"

# ---------------------------------------------------------------- 8. 清理

Step 8 "清理分支（本地 + 远程）"
Run "git branch -d feat/add-gcd"
Run "git push origin --delete feat/add-gcd"
Write-Host ""
Write-Host "剩余分支:" -ForegroundColor Magenta
Run "git branch -a"

# ---------------------------------------------------------------- 9. 模拟"别人更新了，你同步"

Step 9 "模拟上游有新提交，你同步到本地"
Run "git switch -c temp-doc"
Set-Content -Path "NOTES.md" -Value "别人加的笔记" -Encoding UTF8
Run "git add NOTES.md"
Run "git commit -m 'docs: 添加笔记'"
Run "git push -u origin temp-doc"
Run "git switch main"
Run "git merge --no-ff temp-doc -m 'merge: 合入笔记'"
Run "git push origin main"
Run "git branch -d temp-doc"
Run "git push origin --delete temp-doc"
Run "git pull --ff-only origin main"
Run "git log --oneline --graph --decorate"

Pop-Location

Write-Host ""
Write-Host ("=" * 70) -ForegroundColor DarkGray
Write-Host "演练结束。工作目录: $work" -ForegroundColor Green
Write-Host "对应的 GitHub 网页操作：" -ForegroundColor Green
Write-Host "  步骤 5 之后 -> 网页点 'Compare & pull request'"
Write-Host "  步骤 6      -> 网页看 Files changed / Conversation"
Write-Host "  步骤 7      -> 网页点 'Merge pull request'"
Write-Host "  步骤 8      -> 网页点 'Delete branch'"
Write-Host ("=" * 70) -ForegroundColor DarkGray

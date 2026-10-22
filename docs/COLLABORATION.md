# GitHub 协作与 CI

从最新 main 建立个人分支；已有分支每次交付前执行：

```bash
git status --short
git fetch origin
git merge --ff-only origin/feature/a-mlops
git merge origin/main
git merge-base --is-ancestor origin/main HEAD
```

B/C/D 将分支名替换为自己的分支。工作区有未提交修改时先保存，不能 reset、force push 或覆盖队友修改。创建 PR 前再次 fetch；main 有新提交则重新同步并验证。

PR base 固定为 main。验收闭环 A→B→C→D→A：验收人拉取分支运行复现命令后提交 review，通过后 Squash and merge。公共 Schema 和统一入口需要相关负责人追加复核；最终发布 A+B 双签。未经授权不代替验收人合并。

所有成员、所有阶段的验收报告、复核记录及其附件仅保存在本地，不加入 Git、不推送或上传 GitHub；既不能单独提交，也不能随代码提交。新增文件统一放入已忽略的 `.codex_tmp/acceptance/`，提交前检查暂存清单。实验指标、技术方案和比赛要求的镜像材料仍按各自规则保存。

管理员在 GitHub main 分支规则中启用：必须 PR、至少 1 个批准、撤销过期批准、必须解决讨论、必须通过 `checks`、合并前分支保持最新、禁止 force push 和删除。先让 CI 运行一次再选择对应检查；私有仓库是否支持这些规则取决于账户权限。仓库里的工作流不等于已启用远程保护。

CI 使用 Python 3.12 标准库执行编译、单元测试、Shell 语法和合成数据实验两次重跑。它不需要官方数据、GPU、权重或外部模型；完整数据和断网 Docker 验收由负责人本地完成，并由验收人复核。

任务 Issue 应填写负责人、日期、输入依赖、交付文件、验收命令、验收人和状态。阻塞时写明阻塞人、所需接口、预计解除时间和降级方案；状态同步 `docs/EXECUTION_PLAN.md`。

## 分支改名与"会不会覆盖我的改动"

给一个**已有开放 PR** 的分支改名时，`git push <新分支名>` 再 `git push --delete <旧分支名>` 会把 PR 关掉而不是改名（开放 PR 的头分支被删即关闭）。要用 GitHub 的 branch rename 接口，或接受关闭后重开 PR 并重写正文。

评估别人的 PR 会不会回退或覆盖自己的改动时，**不要看两点 diff**（`git diff origin/main..<branch>`）：它是 main 与对方分支的平面比较，不知道三方合并的 base，会把"对方分支只是落后、没有改那几行"显示成"对方把那几行改回了旧值"。用合并模拟读真实结果：

```bash
git merge-base --is-ancestor origin/main <branch>   # 对方是否只是落后
TREE=$(git merge-tree --write-tree origin/main <branch> | head -1)
git show "$TREE:<受关注的文件>"                        # 看合并后的真实内容
```

模拟结果干净（无 CONFLICT）就可以直接合，不必强求对方先 rebase。

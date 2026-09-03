# WSL 9p 文件系统读取延迟

## 现象

监控 cron 每 5 分钟读取 `runs/etl_3cond_v2_gpu_bs128_fp32/best_model.pth`,偶尔读到**过期数据**：

```
实际进度: Epoch 118, val_loss=5.2578
监控读到: Epoch 22, val_loss=6.8052  ← 100 epoch 前的旧值
```

发生频率约 20-30%。epoch checkpoint（`epoch_*.pth`）读取更可靠。

## 根因

WSL2 的 9p 协议文件系统（`/mnt/d/` 挂载 Windows NTFS）存在缓存一致性问题：

- Windows 侧进程（Python）写入 `.pth` 文件
- WSL 侧的读取可能命中**过期的 9p 缓存**
- 文件元数据（`stat` 返回的修改时间）通常是准确的,但文件内容可能滞后
- 频繁覆写的文件（`best_model.pth` 每个 epoch 可能更新）比低频写入的文件（`epoch_N.pth` 只写一次）更容易中招

这不是 Python 的文件缓存,也不是 `torch.save/load` 的 bug——是 9p 协议层面的。

## 缓解措施

1. **交叉验证**: 当 `best_model.pth` 读到可疑旧值时,读 `epoch_*.pth` 确认
2. **加 sleep**: `time.sleep(2)` 后再读,给 9p 缓存同步时间（有一定效果,不完全可靠）
3. **看文件时间戳**: `stat -c '%Y'` 拿修改时间,时间戳通常可信
4. **对监控的影响**: 不影响训练本身（训练进程在 WSL 内部读写,不经过 9p）,只影响从 Windows 侧或另一个 WSL session 的读取

## 不需要修复

这不是数据损坏——训练进程自身的读写路径（同一个 WSL 实例内）是一致的。`best_model.pth` 的实际内容是正确的,只是跨文件系统边界的读取有延迟。

如果后续需要完全可靠的跨系统监控,可以把 checkpoint 写到 WSL 原生文件系统（`~/runs/`）而非 `/mnt/d/`。但当前不值得改——只要知道监控值偶尔过期即可,用 epoch checkpoint 交叉验证就够了。

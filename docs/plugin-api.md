# 云雀笔记 · 插件开发接口

## 插件模型

云雀笔记的插件运行在独立的 Node.js 子进程中，通过 JSON-RPC over stdio 与
主进程通信。这种进程隔离设计意味着插件崩溃不会影响主程序，同时插件也无法
直接访问主进程内存。

每个插件是一个包含 `manifest.json` 与入口文件的目录，安装时会被复制到
`.lark/plugins/<插件 ID>/`。

## manifest.json

```json
{
  "id": "com.example.word-count",
  "name": "字数统计",
  "version": "1.2.0",
  "minAppVersion": "2.0.0",
  "main": "index.js",
  "permissions": ["editor:read", "ui:notice"],
  "activationEvents": ["onCommand:countWords"]
}
```

`permissions` 字段必须显式声明，未声明的能力在调用时会被拒绝并抛出
`PermissionDeniedError`。可用权限包括 `editor:read`、`editor:write`、
`vault:read`、`vault:write`、`network`、`ui:notice`、`ui:panel`。

## 生命周期钩子

| 钩子 | 触发时机 | 超时 |
| --- | --- | --- |
| `onLoad` | 插件加载完成 | 5 秒 |
| `onUnload` | 插件卸载前 | 3 秒 |
| `onNoteOpen` | 打开任意笔记 | 1 秒 |
| `onNoteSave` | 笔记保存后 | 2 秒 |
| `onIndexBuild` | 索引重建完成后 | 10 秒 |
| `onCommand` | 用户执行插件命令 | 30 秒 |

任一钩子超时会被强制终止，连续三次超时后插件会被自动禁用。

## 核心 API

### 读取与修改内容

```javascript
const text = await lark.editor.getText();          // 获取当前笔记全文
await lark.editor.replaceRange(0, 10, "替换内容");   // 替换 [0,10) 区间
await lark.editor.insertAtCursor("插入的文本");
```

### 访问知识库

```javascript
const files = await lark.vault.listFiles({ ext: ".md" });
const content = await lark.vault.read("技术/索引笔记.md");
await lark.vault.write("临时/草稿.md", "# 草稿");
```

`vault:write` 权限只能写入知识库目录内的文件，路径中包含 `..` 或绝对路径
会被拒绝，防止插件越权修改系统文件。

### 网络访问

声明 `network` 权限后可使用 `lark.http.fetch`。该接口默认只允许 HTTPS，
且域名需要在 manifest 的 `allowedDomains` 中白名单声明。

### 界面扩展

```javascript
lark.ui.addCommand({
  id: "countWords",
  name: "统计字数",
  callback: async () => {
    const text = await lark.editor.getText();
    lark.ui.notice(`共 ${text.length} 个字符`);
  }
});
```

## 调试与发布

开发时在"设置 → 插件 → 开发者模式"中开启，即可加载本地未打包的插件目录。
插件子进程的 stdout 会转发到"帮助 → 开发者工具 → 插件日志"面板。

发布前需执行 `lark-plugin validate` 检查 manifest 完整性与权限合理性。
官方插件市场要求插件开源，且不接受声明 `network` 权限但无明确用途说明的
插件。

## 版本兼容

主程序版本号遵循语义化版本。插件声明的 `minAppVersion` 大于当前版本时，
插件会被标记为"不兼容"并禁止加载；`minAppVersion` 小于当前版本两个大版本
时，会显示"可能不兼容"警告但允许加载。

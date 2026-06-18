# Design Context

## Visual Direction

克制、近乎纯灰阶、Codex 风。界面服务任务，不做营销页，不用装饰性效果制造热闹。

## Color

- 使用 OKLCH。
- 禁用纯黑 `#000` 和纯白 `#fff`。
- 背景、中性色微微偏冷，hue 约 280。
- 唯一强调色：`oklch(0.27 0.005 280)`，只用于主按钮、当前步骤、进度。
- 成功色：低饱和青灰 `oklch(0.55 0.02 175)`。

## Typography

- 系统中性无衬线：`-apple-system`, `BlinkMacSystemFont`, `Segoe UI`, `PingFang SC`, `system-ui`, `sans-serif`。
- 不使用展示字体。
- 产品界面使用固定字号，不随视口缩放。

## Layout

- 左侧固定步骤栏。
- 中间主面板一次只显示当前任务。
- 底部操作栏保留唯一主行动。
- 卡片仅用于可选模板、对比输出等明确分组。

## Interaction

- 动效只表达状态变化，150 到 250ms。
- 尊重 `prefers-reduced-motion`。
- 错误提示必须给人话原因和下一步建议。

## Bans

- 不使用侧边色条、渐变文字、玻璃拟态、多个强调色。
- 不把专业训练日志直接暴露为主界面。
- 不用营销式 hero 页面作为首屏。

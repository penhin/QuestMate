# QuestMate Overlay

QuestMate Overlay 是 QuestMate 的 Windows 优先桌面悬浮攻略助手。它使用 Tauri + React + TypeScript 构建悬浮球、小弹窗和可展开右侧抽屉，并调用 FastAPI 后端获取攻略回答。

QuestMate Overlay is the Windows-first desktop floating guide assistant for QuestMate. It uses Tauri + React + TypeScript to provide a floating bubble, compact popover, and expandable right drawer, then calls the FastAPI backend for guide answers.

## 快速开始 / Quick Start

```bash
cp .env.example .env
npm install
npm run dev
```

后端默认地址是 `http://127.0.0.1:8000`，可以通过 `VITE_API_BASE_URL` 修改。

The backend defaults to `http://127.0.0.1:8000`, and can be changed with `VITE_API_BASE_URL`.

## 常用命令 / Useful Commands

```bash
npm run typecheck
npm run build
npm run tauri dev
npm run tauri build
```

## 说明 / Notes

第一版优先支持 Windows 的窗口化和无边框全屏游戏，不承诺覆盖独占全屏。游戏识别会尝试读取前台窗口进程名和标题，识别失败时可以手动输入游戏名。

The first version prioritizes windowed and borderless fullscreen games on Windows, and does not promise exclusive fullscreen overlay support. Game detection tries to read the foreground window process name and title; if detection fails, users can enter the game name manually.

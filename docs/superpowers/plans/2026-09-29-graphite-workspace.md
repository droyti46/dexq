# Graphite Workspace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Реализовать согласованный графитовый интерфейс, проекты текущей вкладки и реальные действия со снимками.

**Architecture:** React provider уровня приложения хранит массив проектов и последовательную очередь анализа. Чистые правила состояния/выделения/CSV тестируются через существующий Node test runner. Workspace и список читают один provider; viewer предоставляет команды меню через ref без глобального event bus.

**Tech Stack:** React 18, TypeScript strict, React Router, Vite; без дополнительных зависимостей.

**Spec:** docs/superpowers/specs/2026-09-29-graphite-workspace-design.md

## Global Constraints
- Только текущая вкладка; перезагрузка/закрытие очищают проекты.
- Не передавать медицинские изображения внешним сервисам.
- Evolventa / Evolventa Bold; собственные компоненты без UI-kit.
- Не изменять модельную логику, не добавлять БД/enterprise patterns.
- Существующие незакоммиченные изменения сохраняются; коммит и push не запрошены.

## Review Focus
- Добавление во время анализа не отменяет прежнюю очередь и не теряет результаты.
- Удаление анализируемого снимка или потокового ZIP не возвращает его после response.
- Назад/вперёд и неизвестный project ID не показывают другой проект по ошибке.
- Ctrl/Shift selection и right/middle drag не конфликтуют с ручными ориентирами.
- Отчёт содержит ошибки, одинаковые имена, кавычки, formula-like имена и не включает queued.

---

### Task 1: Состояние проектов и правила операций
**Files:** create frontend/src/projects.ts, frontend/src/projects.test.ts, frontend/src/components/ProjectProvider.tsx.
**Interfaces:** Project {id,name,createdAt,updatedAt,items,manualAxes,paused}; WorkspaceItem {id,file?,filename,size,position,status,progress,result?,error?,localPreview?,archiveId?}. Provider supplies projects/createProject/addFiles/removeItems/togglePause/updateAxis.
- [ ] Write Node tests for append preserving items, validation/max200, deletion/result no resurrection, range/toggle selection, completed-only CSV with eight columns and safe quoting.
- [ ] Run `node --experimental-strip-types --test frontend/src/projects.test.ts`; expected fail for missing exports.
- [ ] Implement pure add/update/remove/select/export rules; provider owns queue and URL lifecycle. Keep root ZIP while streaming, update only existing children; removing root aborts stream and removes children. Maintain per-project pause, analysis continues across navigation.
- [ ] Run same test command; expected pass.

### Task 2: Проекты, маршруты, меню и workspace
**Files:** modify App.tsx, LandingPage.tsx, WorkspacePage.tsx, Icon.tsx; create ProjectsPage.tsx, ProjectDialog.tsx, DropdownMenu.tsx.
**Interfaces:** Provider Task 1; StudyViewerHandle {zoomIn,zoomOut,fit,reset,toggleOverlay} from Task 3.
- [ ] Browser baseline: /analyze currently opens workspace, Edit has no menu, new files replace old files.
- [ ] Wrap routes in provider, add /projects and /projects/:projectId, redirect old routes to projects. Create project dialog with focus/Escape/backdrop handling; cards include preview stacks and honest empty state.
- [ ] Replace local queue with provider; preserve inspector and progress. Implement additive drop, checkboxes, Ctrl/Cmd/Shift selection, selected actions, delete confirmation, local CSV and project back link.
- [ ] Implement DropdownMenu with outside click/Esc/Arrow/Home/End keyboard handling and disabled items; connect menus to real project/edit/viewer commands.
- [ ] Run build and browser flow create/open/back/select/delete/export; expected no stale project/selection.

### Task 3: Viewer interaction
**Files:** modify StudyViewer.tsx; create viewTransform.ts and viewTransform.test.ts.
**Interfaces:** expose StudyViewerHandle commands to WorkspacePage; overlay state/availability callback for menu.
- [ ] Write zoom math tests: cursor-fixed zoom, offset scaling, clamp 0.25–8, delta direction/limits; run Node test and expect failure before implementation.
- [ ] Implement zoom math, native nonpassive wheel event listener, pointer capture pan on image/background, all right/middle drags, cancel/lost capture cleanup; primary-only point editing. Remove fake orientation indicator (projection unknown).
- [ ] Expose fit/reset/overlay/zoom commands, readable controls/hint, prevent default context menu only in stage. Overlay hides geometry and annotation; reset clears visual adjustment, not medical result/manual axis.
- [ ] Run Node tests and build; browser verify transform/wheel/right/middle/overlay/reset.

### Task 4: Графитовая тема и проверка
**Files:** modify styles.css, docs/user-guide.md; create workspace.css for new/updated workbench styles if it keeps responsibility clear.
- [x] Preserve bright blue public pages per user correction; replace only project/workspace styles with scoped graphite tokens. Remove account/notification styles and obsolete menu-hiding breakpoint.
- [ ] Set readable control sizes, focus, cards/dialog/menu/drop/selection styles, responsive layout retaining menus/inspector/pause, reduced motion.
- [ ] Run all frontend Node tests + npm run build; pytest and ruff from backend. Report any failures accurately.
- [ ] Use preview tools to verify route/menu/project/file/viewer interactions, errors and 375px/768px/desktop. Share screenshot proof.
- [ ] Review code independently where agent access works; if model access unavailable, do explicit self-review and report limitation. Update user guide to current UI and tab-only storage.

## Дополнение пользователя: удобство редактора
- Наложение включено по умолчанию; reset возвращает настройки по умолчанию.
- Выделение рамкой в списке с порогом движения, Ctrl/Cmd добавляет к выделению, автопрокрутка.
- Ручная история по снимкам: одна drag-операция — один undo, redo сбрасывается новым изменением; Ctrl/Cmd+Z, Ctrl+Y, Ctrl/Cmd+Shift+Z, без перехвата полей ввода.
- Приблизительные проценты быстрее подходят к 96%; 100 только при завершении.
- Настоящий Excel .xlsx локально через write-excel-file, отдельный пункт меню; CSV остаётся.
- Один открытый dropdown; hover по соседнему trigger переключает меню, стрелки Left/Right тоже.
- Desktop separators между панелями: pointer resize, клавиатура, doubleclick reset, минимальные ширины и достаточная ширина viewer; на узком экране скрыты.

## Лендинг (дополнительный запрос)
Проморолик frontend/public/media/dexq-demo.mp4 сразу после Hero, на исходном синем фоне. HTML video controls/autoplay/muted/playsInline; autoplay без звука, звук включается пользователем. Hero visual смещается на 32% прокрутки, текст не трансформируется; prefers-reduced-motion отключает эффект.

## Execution ledger
- Plan/spec user-approved in conversation; native execution selected as sensible default because both attempted read-only agents failed with model_not_found.
- No commits/push; no external research upload. Existing backend stream changes remain untouched.
- Final verification: npm run build passed; Node tests 28/28; backend pytest 58 passed, 1 skipped, 3 warnings; Ruff passed; git diff --check passed.
- Browser verified synthetic non-patient fixtures: additive DnD, selection checkbox/Ctrl/Shift/marquee, Delete confirmation, menu hover/keyboard/Esc, wheel cursor zoom, primary/middle/right pan, overlay default/reset, axis keyboard+drag undo/redo (one gesture), local XLSX PK archive export, mobile 375px no overflow, resize 480+480 then1150px reclamps.
- Landing: original blue rgb(2,2,241) preserved; video /media/dexq-demo.mp4 native controls muted autoplay; measured time advancing with no video error. Parallax 32%, reduced-motion guarded.
- Review fixes: incomplete NDJSON stream throws; geometry drawn on native preview; disabled-only menus retain focus; JPEG individual normalized locally; width clamp on viewport resize.
- Deferred: JPEG inside ZIP remains unsupported by backend, explicit UI/docs limitation and separate suggested task. ZIP overflow can retain a root error placeholder in addition to 200 actual images; minor accounting edge, no data loss or >200 actual images.

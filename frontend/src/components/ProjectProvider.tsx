import { createContext, useContext, useEffect, useRef, useState } from 'react';
import type { ReactNode } from 'react';

import { analyzeArchiveIncrementally, analyzeFile } from '../api';
import { commitAxis, undoAxis, redoAxis } from '../axisHistory';
import { estimatedProgress } from '../workspaceInteractions';
import { appendFiles, maxFiles, removeItems, updateItem } from '../projects';
import type { Axis, Project, WorkspaceItem } from '../projects';

interface ProjectContextValue {
  projects: Project[];
  createProject: (name: string) => string;
  addFiles: (projectId: string, files: File[]) => void;
  deleteItems: (projectId: string, ids: string[]) => void;
  togglePause: (projectId: string) => void;
  updateAxis: (projectId: string, itemId: string, axis: Axis | null) => void;
  commitAxisEdit: (projectId: string, itemId: string, before: Axis | null) => void;
  restoreAxis: (projectId: string, itemId: string, direction: 'undo' | 'redo') => void;
}

const ProjectContext = createContext<ProjectContextValue | null>(null);

export function useProjects(): ProjectContextValue {
  const context = useContext(ProjectContext);
  if (!context) throw new Error('ProjectProvider не подключён');
  return context;
}

export default function ProjectProvider({ children }: { children: ReactNode }) {
  const [projects, setProjects] = useState<Project[]>([]);
  const projectsRef = useRef<Project[]>([]);
  const busyRef = useRef(false);
  const controllersRef = useRef(new Map<string, AbortController>());
  const urlsRef = useRef(new Set<string>());
  const mountedRef = useRef(true);

  function changeProject(id: string, change: (project: Project) => Project) {
    if (!mountedRef.current) return;
    projectsRef.current = projectsRef.current.map((project) => project.id === id ? change(project) : project);
    setProjects(projectsRef.current);
  }

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      controllersRef.current.forEach((controller) => controller.abort());
      urlsRef.current.forEach((url) => URL.revokeObjectURL(url));
      urlsRef.current.clear();
    };
  }, []);

  useEffect(() => {
    if (busyRef.current) return;
    const project = projects.find((entry) => !entry.paused && entry.items.some((item) => item.status === 'queued'));
    const item = project?.items.find((entry) => entry.status === 'queued');
    if (!project || !item?.file) return;
    busyRef.current = true;
    const controller = new AbortController();
    controllersRef.current.set(item.id, controller);
    const patch = (id: string, changes: Partial<WorkspaceItem>) => changeProject(project.id, (current) => ({
      ...current, updatedAt: Date.now(), items: updateItem(current.items, id, changes),
    }));
    patch(item.id, { status: 'analyzing', progress: 15 });
    const startedAt = Date.now();
    const timer = window.setInterval(() => {
      changeProject(project.id, (current) => ({ ...current, items: current.items.map((entry) =>
        (entry.id === item.id || entry.archiveId === item.id) && entry.status === 'analyzing'
          ? { ...entry, progress: estimatedProgress(Date.now() - startedAt) } : entry) }));
    }, 150);

    async function process() {
      try {
        if (item!.filename.toLowerCase().endsWith('.zip')) {
          await analyzeArchiveIncrementally(item!.file!, 'auto', (event) => {
            if (controller.signal.aborted) return;
            if (event.type === 'error') throw new Error(event.detail);
            const current = projectsRef.current.find((entry) => entry.id === project!.id);
            if (!current?.items.some((entry) => entry.id === item!.id)) return;
            if (event.type === 'started') {
              if (current.items.length >= maxFiles + 1) throw new Error(`Архив превышает лимит проекта: ${maxFiles} снимков.`);
              const child: WorkspaceItem = {
                id: `${item!.id}:${event.input_position}`, archiveId: item!.id, filename: event.filename,
                size: 0, position: Math.max(0, ...current.items.map((entry) => entry.position)) + 1,
                status: 'analyzing', progress: 18,
              };
              changeProject(project!.id, (entry) => ({ ...entry, items: [...entry.items, child] }));
            }
            if (event.type === 'result') {
              const result = event.item.result ?? undefined;
              patch(`${item!.id}:${event.item.input_position}`, {
                status: result?.processing_status === 'Success' ? 'ready' : 'error', progress: 100, result,
                error: event.item.error ?? result?.error ?? undefined,
              });
            }
          }, controller.signal);
          changeProject(project!.id, (current) => ({ ...current, items: current.items.filter((entry) => entry.id !== item!.id) }));
        } else {
          const result = await analyzeFile(item!.file!, 'auto', controller.signal);
          if (!controller.signal.aborted) patch(item!.id, {
            status: result.processing_status === 'Success' ? 'ready' : 'error', progress: 100, result,
            error: result.error ?? undefined,
          });
        }
      } catch (error) {
        if (!controller.signal.aborted) {
          const message = error instanceof Error ? error.message : 'Не удалось обработать снимок';
          changeProject(project!.id, (current) => ({ ...current, items: current.items.map((entry) =>
            (entry.id === item!.id || entry.archiveId === item!.id) && entry.status === 'analyzing'
              ? { ...entry, status: 'error', progress: 100, error: message } : entry) }));
          controller.abort();
        }
      } finally {
        window.clearInterval(timer);
        controllersRef.current.delete(item!.id);
        busyRef.current = false;
        if (mountedRef.current) setProjects([...projectsRef.current]);
      }
    }
    void process();
  }, [projects]);

  function createProject(name: string): string {
    const id = crypto.randomUUID();
    const now = Date.now();
    projectsRef.current = [{ id, name: name.trim() || 'Новый проект', createdAt: now, updatedAt: now,
      items: [], manualAxes: {}, axisHistories: {}, paused: false }, ...projectsRef.current];
    setProjects(projectsRef.current);
    return id;
  }

  function addFiles(projectId: string, files: File[]) {
    const project = projectsRef.current.find((entry) => entry.id === projectId);
    if (!project) return;
    const next = appendFiles(project.items, files);
    const items = next.map((item) => {
      if (project.items.includes(item) || !item.file || !/\.(png|jpe?g)$/i.test(item.filename)) return item;
      const localPreview = URL.createObjectURL(item.file);
      urlsRef.current.add(localPreview);
      return { ...item, localPreview };
    });
    changeProject(projectId, (current) => ({ ...current, items, updatedAt: Date.now() }));
  }

  function deleteItems(projectId: string, ids: string[]) {
    changeProject(projectId, (current) => {
      const items = removeItems(current.items, ids);
      const removed = current.items.filter((item) => !items.includes(item));
      removed.forEach((item) => {
        controllersRef.current.get(item.id)?.abort();
        if (item.localPreview) {
          URL.revokeObjectURL(item.localPreview);
          urlsRef.current.delete(item.localPreview);
        }
      });
      const manualAxes = { ...current.manualAxes };
      const axisHistories = { ...current.axisHistories };
      removed.forEach((item) => { delete manualAxes[item.id]; delete axisHistories[item.id]; });
      return { ...current, items, manualAxes, axisHistories, updatedAt: Date.now() };
    });
  }

  function togglePause(projectId: string) {
    changeProject(projectId, (project) => ({ ...project, paused: !project.paused }));
  }

  function updateAxis(projectId: string, itemId: string, axis: Axis | null) {
    changeProject(projectId, (project) => {
      const manualAxes = { ...project.manualAxes };
      if (axis) manualAxes[itemId] = axis;
      else delete manualAxes[itemId];
      return { ...project, manualAxes };
    });
  }

  function commitAxisEdit(projectId: string, itemId: string, before: Axis | null) {
    changeProject(projectId, (project) => {
      const history = project.axisHistories?.[itemId] ?? { past: [], present: before, future: [] };
      const committed = commitAxis({ ...history, present: project.manualAxes[itemId] ?? null }, before);
      return { ...project, axisHistories: { ...project.axisHistories, [itemId]: committed } };
    });
  }

  function restoreAxis(projectId: string, itemId: string, direction: 'undo' | 'redo') {
    changeProject(projectId, (project) => {
      const history = project.axisHistories?.[itemId];
      if (!history) return project;
      const next = direction === 'undo' ? undoAxis(history) : redoAxis(history);
      const manualAxes = { ...project.manualAxes };
      if (next.present) manualAxes[itemId] = next.present;
      else delete manualAxes[itemId];
      return { ...project, manualAxes, axisHistories: { ...project.axisHistories, [itemId]: next } };
    });
  }

  return <ProjectContext.Provider value={{ projects, createProject, addFiles, deleteItems, togglePause, updateAxis, commitAxisEdit, restoreAxis }}>
    {children}
  </ProjectContext.Provider>;
}

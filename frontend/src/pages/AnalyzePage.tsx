import { ChangeEvent, DragEvent, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';

import { analyzeStudies } from '../api';
import AppShell from '../components/AppShell';
import type { AnatomicalRegion } from '../types';

const allowedExtensions = ['.dcm', '.dicom', '.png', '.jpg', '.jpeg'];

export default function AnalyzePage() {
  const navigate = useNavigate();
  const inputRef = useRef<HTMLInputElement>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [region, setRegion] = useState<AnatomicalRegion>('auto');
  const [isDragging, setDragging] = useState(false);
  const [isLoading, setLoading] = useState(false);
  const [error, setError] = useState('');

  function acceptFiles(list: FileList | null) {
    if (!list) return;
    const next = Array.from(list).slice(0, 3);
    const invalid = next.find(
      (file) => !allowedExtensions.some((extension) => file.name.toLowerCase().endsWith(extension)),
    );
    if (invalid) {
      setError(`Формат файла ${invalid.name} не поддерживается`);
      return;
    }
    setError('');
    setFiles(next);
  }

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    acceptFiles(event.dataTransfer.files);
  }

  async function submit() {
    if (!files.length) return;
    setLoading(true);
    setError('');
    try {
      const results = await analyzeStudies(files, region);
      sessionStorage.setItem('dexq:last-results', JSON.stringify(results));
      navigate('/result');
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : 'Неизвестная ошибка');
    } finally {
      setLoading(false);
    }
  }

  return (
    <AppShell>
      <main className="page page--analyze">
        <section className="page-intro">
          <p className="eyebrow">Новое исследование</p>
          <h1>Проверка качества DXA</h1>
          <p>Загрузите до трёх изображений одного исследования. Обработка выполняется локально.</p>
        </section>

        <section className="analysis-grid">
          <div className="upload-panel">
            <div
              className={`dropzone ${isDragging ? 'dropzone--active' : ''}`}
              onDragEnter={() => setDragging(true)}
              onDragLeave={() => setDragging(false)}
              onDragOver={(event) => event.preventDefault()}
              onDrop={onDrop}
              onClick={() => inputRef.current?.click()}
              role="button"
              tabIndex={0}
              onKeyDown={(event) => event.key === 'Enter' && inputRef.current?.click()}
            >
              <input
                ref={inputRef}
                type="file"
                accept=".dcm,.dicom,.png,.jpg,.jpeg,application/dicom"
                multiple
                hidden
                onChange={(event: ChangeEvent<HTMLInputElement>) => acceptFiles(event.target.files)}
              />
              <div className="dropzone__icon" aria-hidden="true">
                ↑
              </div>
              <h2>{files.length ? `${files.length} файл(а) выбрано` : 'Перетащите DICOM сюда'}</h2>
              <p>или нажмите, чтобы выбрать на компьютере</p>
              <span>DICOM · PNG/JPEG для демо · до 50 МБ</span>
            </div>

            {files.length > 0 && (
              <ul className="file-list">
                {files.map((file) => (
                  <li key={`${file.name}-${file.size}`}>
                    <div className="file-mark">DX</div>
                    <div>
                      <strong>{file.name}</strong>
                      <span>{(file.size / 1024 / 1024).toFixed(2)} МБ</span>
                    </div>
                    <button
                      type="button"
                      aria-label={`Удалить ${file.name}`}
                      onClick={() => setFiles((current) => current.filter((item) => item !== file))}
                    >
                      ×
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>

          <aside className="settings-card">
            <span className="step-number">02</span>
            <h2>Параметры анализа</h2>
            <label htmlFor="region">Анатомическая область</label>
            <select
              id="region"
              value={region}
              onChange={(event) => setRegion(event.target.value as AnatomicalRegion)}
            >
              <option value="auto">Определить автоматически</option>
              <option value="lumbar_spine">Поясничный отдел</option>
              <option value="proximal_femur">Проксимальный отдел бедра</option>
            </select>
            <div className="privacy-note">
              <span aria-hidden="true">⌁</span>
              <p>
                Файлы не сохраняются. Персональные DICOM-теги не попадают в отчёт.
              </p>
            </div>
            {error && <p className="form-error">{error}</p>}
            <button
              className="button button--primary button--full"
              type="button"
              disabled={!files.length || isLoading}
              onClick={submit}
            >
              {isLoading ? 'Выполняем проверку…' : 'Начать анализ'}
            </button>
          </aside>
        </section>
      </main>
    </AppShell>
  );
}


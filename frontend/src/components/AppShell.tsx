import type { ReactNode } from 'react';
import { Link, NavLink } from 'react-router-dom';

interface AppShellProps {
  children: ReactNode;
  dark?: boolean;
}

export default function AppShell({ children, dark = false }: AppShellProps) {
  return (
    <div className={dark ? 'app-shell app-shell--dark' : 'app-shell'}>
      <header className="site-header">
        <Link className="wordmark" to="/" aria-label="DEXQ — главная">
          DEXQ<span className="wordmark__cut" />
        </Link>
        <nav className="site-nav" aria-label="Основная навигация">
          <NavLink to="/about">О продукте</NavLink>
          <a href="http://localhost:8000/docs" target="_blank" rel="noreferrer">
            API
          </a>
          <NavLink className="nav-cta" to="/analyze">
            Анализ
          </NavLink>
        </nav>
      </header>
      {children}
    </div>
  );
}


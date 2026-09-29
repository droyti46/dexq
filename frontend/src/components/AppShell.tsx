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
        <nav className="site-nav" aria-label="Основная навигация">
          <a href="https://droyti46.github.io/dexq/" target="_blank" rel="noreferrer">
            Документация
          </a>
          <Link className="wordmark" to="/" aria-label="DEXQ — главная">
            DEXQ<span className="wordmark__cut" />
          </Link>
          <NavLink to="/team">
            Наша команда
          </NavLink>
        </nav>
      </header>
      {children}
    </div>
  );
}

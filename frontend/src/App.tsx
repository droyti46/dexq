import { Navigate, Route, Routes } from 'react-router-dom';

import AboutPage from './pages/AboutPage';
import LandingPage from './pages/LandingPage';
import TeamPage from './pages/TeamPage';
import WorkspacePage from './pages/WorkspacePage';
import ProjectsPage from './pages/ProjectsPage';
import ProjectProvider from './components/ProjectProvider';

export default function App() {
  return (
    <ProjectProvider><Routes>
      <Route path="/" element={<LandingPage />} />
      <Route path="/projects" element={<ProjectsPage />} />
      <Route path="/projects/:projectId" element={<WorkspacePage />} />
      <Route path="/analyze" element={<Navigate to="/projects" replace />} />
      <Route path="/result" element={<Navigate to="/projects" replace />} />
      <Route path="/about" element={<AboutPage />} />
      <Route path="/team" element={<TeamPage />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes></ProjectProvider>
  );
}

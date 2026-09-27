import { Navigate, Route, Routes } from 'react-router-dom';

import AboutPage from './pages/AboutPage';
import AnalyzePage from './pages/AnalyzePage';
import LandingPage from './pages/LandingPage';
import ResultPage from './pages/ResultPage';

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route path="/analyze" element={<AnalyzePage />} />
      <Route path="/result" element={<ResultPage />} />
      <Route path="/about" element={<AboutPage />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}


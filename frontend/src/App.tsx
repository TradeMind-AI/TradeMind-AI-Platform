import {
  BrowserRouter,
  Route,
  Routes,
} from 'react-router-dom'

import Dashboard
  from './pages/Dashboard/Dashboard'

import Landing
  from './pages/Landing/Landing'

import SignalsPage
  from './pages/Signals/Signals'

import PerformancePage
  from './pages/Performance/Performance'

import ModelPage
  from './pages/Model/Model'

import RiskPage
  from './pages/Risk/Risk'

import ForwardValidationPage
  from './pages/ForwardValidation/ForwardValidation'


function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route
          path="/"
          element={<Landing />}
        />

        <Route
          path="/dashboard"
          element={<Dashboard />}
        />

        <Route
          path="/dashboard/signals"
          element={<SignalsPage />}
        />

        <Route
          path="/dashboard/performance"
          element={<PerformancePage />}
        />

        <Route
          path="/dashboard/model"
          element={<ModelPage />}
        />

        <Route
          path="/dashboard/risk"
          element={<RiskPage />}
        />

        <Route
          path="/dashboard/forward-validation"
          element={<ForwardValidationPage />}
        />
      </Routes>
    </BrowserRouter>
  )
}


export default App

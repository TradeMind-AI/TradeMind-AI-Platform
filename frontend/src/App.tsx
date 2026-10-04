import {
  BrowserRouter,
  Route,
  Routes,
} from 'react-router-dom'

import Dashboard from './pages/Dashboard/Dashboard'
import Landing from './pages/Landing/Landing'

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
      </Routes>
    </BrowserRouter>
  )
}

export default App
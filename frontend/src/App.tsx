import {
  BrowserRouter,
  Route,
  Routes,
} from 'react-router-dom'

import Dashboard from './pages/Dashboard/Dashboard'
import Landing from './pages/Landing/Landing'
import Forex from './pages/Forex/Forex'

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
        <Route path="/forex" element={<Forex />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
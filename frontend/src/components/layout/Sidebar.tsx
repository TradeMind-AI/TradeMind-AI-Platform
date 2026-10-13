import {
  NavLink,
} from 'react-router-dom'


type SidebarItemProps = {
  label: string
  to: string
}


const SidebarItem = ({
  label,
  to,
}: SidebarItemProps) => {
  return (
    <NavLink
      to={to}
      className={({ isActive }) => [
        'block w-full rounded-lg px-4 py-3 text-left text-sm transition',
        isActive
          ? 'bg-white/10 text-white'
          : 'text-white/40 hover:bg-white/[0.04] hover:text-white',
      ].join(' ')}
    >
      {label}
    </NavLink>
  )
}


const Sidebar = () => {
  return (
    <aside className="hidden min-h-screen w-64 border-r border-white/5 bg-[#050914] px-5 py-6 lg:block">
      <NavLink
        to="/"
        className="flex items-center gap-3 px-2"
      >
        <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-white/10 bg-white/5">
          <span className="font-semibold">
            T
          </span>
        </div>

        <div>
          <p className="font-semibold">
            TradeMind AI
          </p>

          <p className="text-xs text-white/30">
            Trading Intelligence
          </p>
        </div>
      </NavLink>

      <nav className="mt-10 space-y-2">
        <SidebarItem
          label="Overview"
          to="/dashboard"
        />

        <SidebarItem
          label="Signals"
          to="/dashboard/signals"
        />

        <SidebarItem
          label="Performance"
          to="/dashboard/performance"
        />

        <SidebarItem
          label="Model"
          to="/dashboard/model"
        />

        <SidebarItem
          label="Risk"
          to="/dashboard/risk"
        />
      </nav>

      <div className="mt-10 rounded-xl border border-white/5 bg-white/[0.02] p-4">
        <p className="text-xs uppercase tracking-[0.2em] text-white/20">
          Mode
        </p>

        <p className="mt-2 text-sm font-medium text-white/70">
          Paper Trading
        </p>
      </div>
    </aside>
  )
}


export default Sidebar

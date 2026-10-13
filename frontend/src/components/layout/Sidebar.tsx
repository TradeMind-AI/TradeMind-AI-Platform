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
          ? 'border border-[#d4af37]/25 bg-[#d4af37]/10 text-[#f0cf68] shadow-[inset_3px_0_0_#d4af37]'
          : 'border border-transparent text-white/40 hover:border-[#d4af37]/10 hover:bg-[#d4af37]/[0.04] hover:text-[#e7d8ad]',
      ].join(' ')}
    >
      {label}
    </NavLink>
  )
}


const Sidebar = () => {
  return (
    <aside className="hidden min-h-screen w-64 border-r border-[#d4af37]/10 bg-[#050505] px-5 py-6 lg:block">
      <NavLink
        to="/"
        className="flex items-center gap-3 px-2"
      >
        <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-[#d4af37]/35 bg-[#d4af37]/[0.06] shadow-[0_0_24px_rgba(212,175,55,0.08)]">
          <span className="font-semibold text-[#f0cf68]">
            T
          </span>
        </div>

        <div>
          <p className="font-semibold">
            TradeMind AI
          </p>

          <p className="text-xs text-[#9d8650]">
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

        <SidebarItem
          label="Forward Validation"
          to="/dashboard/forward-validation"
        />
      </nav>

      <div className="mt-10 rounded-xl border border-[#d4af37]/15 bg-[#d4af37]/[0.025] p-4">
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

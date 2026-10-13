import type {
  ReactNode,
} from 'react'

import Sidebar
  from './Sidebar'


type DashboardPageShellProps = {
  eyebrow: string
  title: string
  description: string
  children: ReactNode
}


const DashboardPageShell = ({
  eyebrow,
  title,
  description,
  children,
}: DashboardPageShellProps) => {
  return (
    <div className="min-h-screen bg-[#030712] text-white">
      <div className="flex">
        <Sidebar />

        <div className="min-w-0 flex-1">
          <header className="border-b border-white/5 px-6 py-6 lg:px-8">
            <div className="mx-auto max-w-7xl">
              <p className="text-xs uppercase tracking-[0.2em] text-white/25">
                TradeMind AI
              </p>

              <h1 className="mt-2 text-2xl font-semibold">
                {title}
              </h1>
            </div>
          </header>

          <main className="px-6 py-8 lg:px-8">
            <div className="mx-auto max-w-7xl">
              <div>
                <p className="text-xs uppercase tracking-[0.2em] text-white/30">
                  {eyebrow}
                </p>

                <h2 className="mt-2 text-3xl font-semibold tracking-tight">
                  {title}
                </h2>

                <p className="mt-3 max-w-3xl text-sm leading-6 text-white/35">
                  {description}
                </p>
              </div>

              <div className="mt-8">
                {children}
              </div>
            </div>
          </main>
        </div>
      </div>
    </div>
  )
}


export default DashboardPageShell

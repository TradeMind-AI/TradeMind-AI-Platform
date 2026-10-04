type StatCardProps = {
  label: string
  value: string
  subtext?: string
}

const StatCard = ({
  label,
  value,
  subtext,
}: StatCardProps) => {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.025] p-5">
      <p className="text-sm text-white/35">
        {label}
      </p>

      <p className="mt-3 text-2xl font-semibold">
        {value}
      </p>

      {subtext && (
        <p className="mt-2 text-xs text-white/25">
          {subtext}
        </p>
      )}
    </div>
  )
}

export default StatCard
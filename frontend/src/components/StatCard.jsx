function StatCard({ title, value }) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900 p-5">

      <p className="text-sm text-slate-400">
        {title}
      </p>

      <p className="mt-2 text-2xl font-semibold">
        {value}
      </p>

    </div>
  );
}

export default StatCard;
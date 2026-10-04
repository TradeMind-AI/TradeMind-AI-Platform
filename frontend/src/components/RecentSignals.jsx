function RecentSignals({ signals }) {
  return (
    <div className="overflow-hidden rounded-xl border border-slate-800 bg-slate-900">

      <div className="border-b border-slate-800 px-6 py-4">

        <h3 className="text-lg font-semibold">
          Recent Signals
        </h3>

      </div>

      <div className="overflow-x-auto">

        <table className="w-full text-left">

          <thead className="bg-slate-950 text-sm text-slate-400">

            <tr>

              <th className="px-6 py-3">
                Time
              </th>

              <th className="px-6 py-3">
                Prediction
              </th>

              <th className="px-6 py-3">
                Confidence
              </th>

              <th className="px-6 py-3">
                Signal
              </th>

            </tr>

          </thead>

          <tbody>

            {signals.map(
              (signal, index) => (
                <tr
                  key={index}
                  className="border-t border-slate-800"
                >

                  <td className="px-6 py-4 text-sm">
                    {signal.time}
                  </td>

                  <td className="px-6 py-4">
                    {signal.prediction}
                  </td>

                  <td className="px-6 py-4">
                    {signal.confidence}
                  </td>

                  <td className="px-6 py-4 font-semibold">
                    {signal.signal}
                  </td>

                </tr>
              )
            )}

          </tbody>

        </table>

      </div>

    </div>
  );
}

export default RecentSignals;
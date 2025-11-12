import { useMemo, useState } from 'react'
import './App.css'

const algorithms = [
  { id: 'depth-limited', label: 'Depth-Limited Huffman' },
  { id: 'adaptive', label: 'Adaptive Huffman' },
]

function App() {
  const [selectedAlgorithm, setSelectedAlgorithm] = useState(algorithms[0].id)
  const [lmax, setLmax] = useState(4)
  const disabledMetrics = useMemo(
    () => [
      { label: 'avgCodeLen', value: '--' },
      { label: 'compressionRatio', value: '--' },
    ],
    [],
  )

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <h1>Huffman Visualizer</h1>
          <p>Configure the algorithm, upload a dataset, and watch the tree build itself.</p>
        </div>
        <button className="primary-btn">Repository</button>
      </header>

      <section className="panel controls-panel">
        <div className="control-group">
          <label htmlFor="algorithm">Algorithm</label>
          <select
            id="algorithm"
            value={selectedAlgorithm}
            onChange={(event) => setSelectedAlgorithm(event.target.value)}
          >
            {algorithms.map((algorithm) => (
              <option key={algorithm.id} value={algorithm.id}>
                {algorithm.label}
              </option>
            ))}
          </select>
        </div>

        <div className="control-group">
          <label htmlFor="dataset">Dataset</label>
          <label className="file-input">
            <input id="dataset" type="file" />
            <span>Select file</span>
          </label>
        </div>

        <div className="control-group slider-group">
          <label htmlFor="lmax">Lmax</label>
          <div className="slider-with-value">
            <input
              id="lmax"
              type="range"
              min={1}
              max={8}
              step={1}
              value={lmax}
              onChange={(event) => setLmax(Number(event.target.value))}
            />
            <span>{lmax}</span>
          </div>
        </div>

        <div className="control-group run-group">
          <button className="primary-btn">Run</button>
        </div>
      </section>

      <div className="content-grid">
        <section className="panel tree-panel">
          <div className="panel-header">
            <h2>Animated Tree</h2>
            <span className="panel-subtitle">Preview of the generated Huffman tree</span>
          </div>
          <div className="tree-canvas">
            <div className="tree-node tree-node-root">root</div>
            <div className="tree-branch tree-branch-left" />
            <div className="tree-branch tree-branch-right" />
            <div className="tree-node tree-node-left">A</div>
            <div className="tree-node tree-node-right">B</div>
          </div>
        </section>

        <aside className="side-column">
          <section className="panel metrics-panel">
            <div className="panel-header">
              <h2>Metrics</h2>
              <span className="panel-subtitle">Live metrics update as the tree evolves</span>
            </div>
            <div className="metric-cards">
              {disabledMetrics.map((metric) => (
                <article key={metric.label} className="metric-card">
                  <span className="metric-label">{metric.label}</span>
                  <span className="metric-value">{metric.value}</span>
                </article>
              ))}
            </div>
          </section>

          <section className="panel charts-panel">
            <div className="panel-header">
              <h2>Charts</h2>
              <span className="panel-subtitle">Compare against baseline algorithms</span>
            </div>
            <div className="chart-stack">
              <div className="chart-placeholder">
                <span>avgCodeLen vs baseline</span>
              </div>
              <div className="chart-placeholder">
                <span>compressionRatio vs baseline</span>
              </div>
            </div>
          </section>
        </aside>
      </div>
    </div>
  )
}

export default App
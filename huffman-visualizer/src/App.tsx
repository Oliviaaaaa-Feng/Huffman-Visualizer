import { useMemo, useState, useEffect } from 'react'
import './App.css'

const API_BASE = 'http://localhost:8000'

type TreeNode = {
  id: string
  weight: number
  depth: number
  left?: string | null
  right?: string | null
}

type AdaptiveStep = {
  step: number
  symbol: number | string
  action: string
  tree: TreeNode[]
}

const algorithms = [
  { id: 'depth-limited', label: 'Depth-Limited Huffman' },
  { id: 'adaptive', label: 'Adaptive Huffman' },
]

function App() {
  const [selectedAlgorithm, setSelectedAlgorithm] = useState(algorithms[0].id)

  // depth-limited state
  const [lmax, setLmax] = useState(4)
  const [probabilities, setProbabilities] = useState('[0.4,0.3,0.2,0.1]')
  const [tree, setTree] = useState<TreeNode[] | null>(null)

  // adaptive state
  const [adaptiveText, setAdaptiveText] = useState('ABRACADABRA')
  const [adaptiveSteps, setAdaptiveSteps] = useState<AdaptiveStep[] | null>(null)
  const [adaptiveCurrentStepIndex, setAdaptiveCurrentStepIndex] = useState(0)
  const [adaptiveBitstream, setAdaptiveBitstream] = useState<string>('')
  const [isAdaptivePlaying, setIsAdaptivePlaying] = useState(false)

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [datasetFile, setDatasetFile] = useState<File | null>(null)

  const disabledMetrics = useMemo(
    () => [
      { label: 'avgCodeLen', value: '--' },
      { label: 'compressionRatio', value: '--' },
    ],
    [],
  )

  const run = async () => {
    setLoading(true)
    setError(null)

    // reset state for both modes
    setTree(null)
    setAdaptiveSteps(null)
    setAdaptiveCurrentStepIndex(0)
    setAdaptiveBitstream('')
    setIsAdaptivePlaying(false)

    try {
      if (selectedAlgorithm === 'depth-limited') {
        const form = new FormData()
        form.append('probabilities', probabilities)
        form.append('lmax', String(lmax))

        const res = await fetch(`${API_BASE}/api/limited-depth/preview`, {
          method: 'POST',
          body: form,
        })
        const data = await res.json()

        if (!res.ok || data.error) {
          throw new Error(data.error || `HTTP ${res.status}`)
        }

        setTree(data.tree as TreeNode[])
      } else if (selectedAlgorithm === 'adaptive') {
        const form = new FormData()
        form.append('text', adaptiveText)

        const res = await fetch(`${API_BASE}/api/adaptive/preview`, {
          method: 'POST',
          body: form,
        })
        const data = await res.json()

        if (!res.ok || data.error) {
          throw new Error(data.error || `HTTP ${res.status}`)
        }

        const steps = (data.steps || []) as AdaptiveStep[]
        setAdaptiveSteps(steps)
        setAdaptiveCurrentStepIndex(0)
        setAdaptiveBitstream(data.bitstream || '')
        setIsAdaptivePlaying(true)
      } else {
        throw new Error('Unknown algorithm')
      }
    } catch (e: any) {
      setError(e.message || 'Request failed')
    } finally {
      setLoading(false)
    }
  }

  const isDepthLimited = selectedAlgorithm === 'depth-limited'
  const isAdaptive = selectedAlgorithm === 'adaptive'

  useEffect(() => {
    if (
      !isAdaptive ||
      !adaptiveSteps ||
      adaptiveSteps.length === 0 ||
      !isAdaptivePlaying
    )
      return

    const total = adaptiveSteps.length
    const intervalId = window.setInterval(() => {
      setAdaptiveCurrentStepIndex((idx) => {
        if (idx >= total - 1) {
          window.clearInterval(intervalId)
          setIsAdaptivePlaying(false)
          return total - 1
        }
        return idx + 1
      })
    }, 900)

    return () => {
      window.clearInterval(intervalId)
    }
  }, [isAdaptive, adaptiveSteps, isAdaptivePlaying])

  const currentAdaptiveStep =
    isAdaptive && adaptiveSteps && adaptiveSteps.length > 0
      ? adaptiveSteps[Math.min(adaptiveCurrentStepIndex, adaptiveSteps.length - 1)]
      : null

  const previousAdaptiveStep =
    isAdaptive &&
    adaptiveSteps &&
    adaptiveSteps.length > 0 &&
    adaptiveCurrentStepIndex > 0
      ? adaptiveSteps[adaptiveCurrentStepIndex - 1]
      : null

  const goToPrevStep = () => {
    if (!adaptiveSteps) return
    setIsAdaptivePlaying(false)
    setAdaptiveCurrentStepIndex((idx) => Math.max(0, idx - 1))
  }

  const goToNextStep = () => {
    if (!adaptiveSteps) return
    setIsAdaptivePlaying(false)
    setAdaptiveCurrentStepIndex((idx) =>
      Math.min(adaptiveSteps.length - 1, idx + 1),
    )
  }

  const goToStep = (idx: number) => {
    if (!adaptiveSteps) return
    setIsAdaptivePlaying(false)
    setAdaptiveCurrentStepIndex(idx)
  }

  const replayAdaptive = () => {
    if (!adaptiveSteps || adaptiveSteps.length === 0) return
    setAdaptiveCurrentStepIndex(0)
    setIsAdaptivePlaying(true)
  }

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <h1>Huffman Visualizer</h1>
          <p>Configure the algorithm, upload a dataset, and watch the tree build itself.</p>
        </div>
        <button
          className="primary-btn"
          onClick={() =>
            window.open(
              'https://github.com/Oliviaaaaa-Feng/Huffman-Visualizer',
              '_blank',
              'noopener,noreferrer'
            )
          }
        >
          Repository
        </button>

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
            <input
              id="dataset"
              type="file"
              onChange={(event) => {
                const file = event.target.files?.[0] ?? null
                setDatasetFile(file)
              }}
            />
            <span>{datasetFile ? datasetFile.name : 'Select file'}</span>
          </label>
        </div>

        {isDepthLimited && (
          <>
            <div className="control-group">
              <label htmlFor="probabilities">Probabilities</label>
              <input
                id="probabilities"
                type="text"
                placeholder='[0.4,0.3,0.2,0.1]'
                value={probabilities}
                onChange={(event) => setProbabilities(event.target.value)}
              />
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
          </>
        )}

        {isAdaptive && (
          <div className="control-group">
            <label htmlFor="adaptiveText">Input text</label>
            <input
              id="adaptiveText"
              type="text"
              placeholder="ABRACADABRA"
              value={adaptiveText}
              onChange={(event) => setAdaptiveText(event.target.value)}
            />
          </div>
        )}

        <div className="control-group run-group">
          <button className="primary-btn" onClick={run} disabled={loading}>
            {loading ? 'Running…' : 'Run'}
          </button>
        </div>
      </section>

      <div className="content-grid">
        <section className="panel tree-panel">
          <div className="panel-header">
            <h2>Tree View</h2>
            <span className="panel-subtitle">
              {isDepthLimited
                ? 'Huffman tree built from the given probability distribution'
                : 'Adaptive Huffman tree as it evolves over time'}
            </span>
          </div>

          <div className="tree-canvas">
            {error && <div className="error">{error}</div>}
            {!error && isDepthLimited && !tree && (
              <div className="placeholder">Click Run to build the tree</div>
            )}
            {!error && isDepthLimited && tree && <TreeSVG nodes={tree} />}
            {!error && isAdaptive && !adaptiveSteps && (
              <div className="placeholder">
                Enter a short text (e.g., ABRACADABRA) and click Run
              </div>
            )}
            {!error && isAdaptive && adaptiveSteps && currentAdaptiveStep && (
              <AdaptiveTreeSVG
                step={currentAdaptiveStep}
                previousStep={previousAdaptiveStep}
              />
            )}
          </div>

          {isAdaptive && adaptiveSteps && adaptiveSteps.length > 0 && currentAdaptiveStep && (
            <>
              <div className="tree-steps">
                <button className="ghost-btn" onClick={goToPrevStep}>
                  ◀ Prev
                </button>

                <span className="step-label">
                  Step {adaptiveCurrentStepIndex + 1} / {adaptiveSteps.length}
                </span>


                <button className="ghost-btn" onClick={goToNextStep}>
                  Next ▶
                </button>

                <button
                  className="ghost-btn"
                  onClick={replayAdaptive}
                  style={{ marginLeft: '12px' }}
                >
                  {isAdaptivePlaying ? 'Playing…' : 'Replay'}
                </button>
              </div>

              <div className="step-list">
                {adaptiveSteps.map((s, idx) => (
                  <button
                    key={s.step}
                    className={
                      idx === adaptiveCurrentStepIndex
                        ? 'step-chip step-chip-active'
                        : 'step-chip'
                    }
                    onClick={() => goToStep(idx)}
                  >
                    {idx + 1}
                  </button>
                ))}
              </div>

              {adaptiveBitstream && (
                <div className="bitstream-box">
                  <div className="bitstream-label">Encoded bitstream</div>
                  <div className="bitstream-content">
                    <code>{adaptiveBitstream}</code>
                  </div>
                </div>
              )}
            </>
          )}
        </section>

        <aside className="side-column">
          <section className="panel metrics-panel">
            <div className="panel-header">
              <h2>Metrics</h2>
              <span className="panel-subtitle">
                (Placeholder) Metrics will go here later
              </span>
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
              <span className="panel-subtitle">
                (Placeholder) Compare against baseline algorithms
              </span>
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

function TreeSVG({ nodes }: { nodes: TreeNode[] }) {
  const nodeMap = useMemo(() => {
    const m = new Map<string, TreeNode>()
    nodes.forEach((n) => m.set(n.id, n))
    return m
  }, [nodes])

  const root = useMemo(() => {
    return nodes.find((n) => n.id === '*') ?? nodes.find((n) => n.depth === 0)!
  }, [nodes])

  function getLeaves(id: string): string[] {
    const node = nodeMap.get(id)!
    if (!node.left && !node.right) return [id]
    let res: string[] = []
    if (node.left) res = res.concat(getLeaves(node.left))
    if (node.right) res = res.concat(getLeaves(node.right))
    return res
  }

  const leaves = getLeaves(root.id)
  const leafX = new Map<string, number>()
  leaves.forEach((leaf, i) => leafX.set(leaf, i))

  function computeX(id: string): number {
    const node = nodeMap.get(id)!
    if (!node.left && !node.right) return leafX.get(id)!
    const lx = computeX(node.left!)
    const rx = computeX(node.right!)
    return (lx + rx) / 2
  }

  const positions = new Map<string, { x: number; y: number }>()
  const verticalGap = 140
  const horizontalScale = 150

  function assignPos(id: string, depth: number) {
    const x = computeX(id) * horizontalScale + 80
    const y = depth * verticalGap + 60
    positions.set(id, { x, y })
    const node = nodeMap.get(id)!
    if (node.left) assignPos(node.left, depth + 1)
    if (node.right) assignPos(node.right, depth + 1)
  }

  assignPos(root.id, 0)

  const edges: Array<{ from: string; to: string }> = []
  nodes.forEach((n) => {
    if (n.left) edges.push({ from: n.id, to: n.left })
    if (n.right) edges.push({ from: n.id, to: n.right })
  })

  const width = leaves.length * horizontalScale + 160
  const height =
    (Math.max(...nodes.map((n) => n.depth)) + 1) * verticalGap + 100

  const boxW = 90
  const boxH = 54
  const radius = 12

  return (
    <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`}>
      {edges.map((e, i) => {
        const p = positions.get(e.from)!
        const c = positions.get(e.to)!
        const startY = p.y + boxH / 2
        const endY = c.y - boxH / 2
        const midY = (startY + endY) / 2
        const d = `M ${p.x} ${startY}
                   C ${p.x} ${midY},
                     ${c.x} ${midY},
                     ${c.x} ${endY}`

        return (
          <path
            key={i}
            d={d}
            fill="none"
            stroke="#45465c"
            strokeWidth={2}
            strokeLinecap="round"
          />
        )
      })}

      {nodes.map((n) => {
        const pos = positions.get(n.id)!
        const isLeaf = !n.left && !n.right

        const bgFill = isLeaf ? '#e2d6ff' : '#e6e6ef'
        const border = '#9a8bff'

        return (
          <g
            key={n.id}
            transform={`translate(${pos.x}, ${pos.y})`}
            textAnchor="middle"
          >
            <rect
              x={-boxW / 2}
              y={-boxH / 2}
              width={boxW}
              height={boxH}
              rx={radius}
              ry={radius}
              fill={bgFill}
              stroke={border}
              strokeWidth={2}
            />

            <text y={-4} fontSize={14} fontWeight={600} fill="#222339">
              {n.id}
            </text>

            <text y={14} fontSize={12} fill="#555976">
              {`p=${n.weight.toFixed(2)}`}
            </text>
          </g>
        )
      })}
    </svg>
  )
}

function AdaptiveTreeSVG({
  step,
  previousStep,
}: {
  step: AdaptiveStep
  previousStep: AdaptiveStep | null
}) {
  const nodes = step.tree

  // nodes that already existed in previous step
  const previousIds = useMemo(() => {
    if (!previousStep) return new Set<string>()
    return new Set(previousStep.tree.map((n) => n.id))
  }, [previousStep])

  // previous step weights
  const previousWeights = useMemo(() => {
    const m = new Map<string, number>()
    if (!previousStep) return m
    previousStep.tree.forEach((n) => {
      m.set(n.id, n.weight)
    })
    return m
  }, [previousStep])

  // id -> node
  const nodeMap = useMemo(() => {
    const m = new Map<string, TreeNode>()
    nodes.forEach((n) => m.set(n.id, n))
    return m
  }, [nodes])

  // root node
  const root = useMemo(() => {
    return nodes.find((n) => n.id === '*') ?? nodes.find((n) => n.depth === 0)!
  }, [nodes])

  // children sorted by weight (small -> large) for layout
  function getChildrenSorted(id: string): string[] {
    const node = nodeMap.get(id)!
    const children: string[] = []
    if (node.left) children.push(node.left)
    if (node.right) children.push(node.right)
    children.sort((aId, bId) => {
      const aw = nodeMap.get(aId)!.weight
      const bw = nodeMap.get(bId)!.weight
      return aw - bw
    })
    return children
  }

  // collect leaves in left-to-right order based on sorted children
  function collectLeaves(id: string, acc: string[]) {
    const children = getChildrenSorted(id)
    if (children.length === 0) {
      acc.push(id)
      return
    }
    for (const child of children) {
      collectLeaves(child, acc)
    }
  }

  const leaves: string[] = []
  collectLeaves(root.id, leaves)

  const leafX = new Map<string, number>()
  leaves.forEach((leaf, i) => leafX.set(leaf, i))

  // compute x-position using leaf order
  function computeX(id: string): number {
    const node = nodeMap.get(id)!
    const children = getChildrenSorted(id)
    if (children.length === 0) return leafX.get(id)!
    const xs = children.map((childId) => computeX(childId))
    const minX = Math.min(...xs)
    const maxX = Math.max(...xs)
    return (minX + maxX) / 2
  }

  const positions = new Map<string, { x: number; y: number }>()
  const verticalGap = 140
  const horizontalScale = 150

  function assignPos(id: string, depth: number) {
    const x = computeX(id) * horizontalScale + 80
    const y = depth * verticalGap + 60
    positions.set(id, { x, y })
    const children = getChildrenSorted(id)
    for (const childId of children) {
      assignPos(childId, depth + 1)
    }
  }

  assignPos(root.id, 0)

  // build edges using sorted children so they match the layout
  const edges: Array<{ from: string; to: string }> = []
  nodes.forEach((n) => {
    const children = getChildrenSorted(n.id)
    children.forEach((childId) => {
      edges.push({ from: n.id, to: childId })
    })
  })

  const width = leaves.length * horizontalScale + 160
  const height =
    (Math.max(...nodes.map((n) => n.depth)) + 1) * verticalGap + 100

  const boxW = 90
  const boxH = 54
  const radius = 12

  return (
    <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`}>
      {/* edges */}
      {edges.map((e, i) => {
        const p = positions.get(e.from)!
        const c = positions.get(e.to)!
        const startY = p.y + boxH / 2
        const endY = c.y - boxH / 2
        const midY = (startY + endY) / 2
        const d = `M ${p.x} ${startY}
                   C ${p.x} ${midY},
                     ${c.x} ${midY},
                     ${c.x} ${endY}`

        return (
          <path
            key={i}
            d={d}
            fill="none"
            stroke="#45465c"
            strokeWidth={2}
            strokeLinecap="round"
          />
        )
      })}

      {/* nodes */}
      {nodes.map((n) => {
        const pos = positions.get(n.id)!
        const isLeaf = !n.left && !n.right

        const isNewNode = !previousIds.has(n.id)
        const prevW = previousWeights.get(n.id)
        const isUpdated = prevW !== undefined && prevW !== n.weight
        const bgFill = isLeaf
          ? (isNewNode ? '#f5e2ff' : '#e2d6ff')
          : (isNewNode ? '#eef0ff' : '#e6e6ef')
        const border = isNewNode ? '#ff7ab5' : '#9a8bff'
        const weightColor = isUpdated ? '#ff3b8b' : '#555976'
        const weightFontWeight = isUpdated ? 700 : 400

        return (
          <g
            key={n.id}
            transform={`translate(${pos.x}, ${pos.y})`}
            textAnchor="middle"
          >
            <rect
              x={-boxW / 2}
              y={-boxH / 2}
              width={boxW}
              height={boxH}
              rx={radius}
              ry={radius}
              fill={bgFill}
              stroke={border}
              strokeWidth={2}
            />

            {/* Node ID */}
            <text y={-4} fontSize={14} fontWeight={600} fill="#222339">
              {n.id}
            </text>

            <text
              y={14}
              fontSize={12}
              fill={weightColor}
              fontWeight={weightFontWeight}
            >
              {`w=${n.weight.toFixed(0)}`}
            </text>
          </g>
        )
      })}
    </svg>
  )
}

export default App

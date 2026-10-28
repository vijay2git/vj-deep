
import { useEffect, useState } from 'react'
import {
  Activity,
  BrainCircuit,
  CheckCircle2,
  CircleDot,
  Clock3,
  Gauge,
  HeartPulse,
  LayoutDashboard,
  Radio,
  RefreshCw,
  Settings2,
  ShieldCheck,
  Waves,
} from 'lucide-react'
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import './index.css'

const API_URL = 'http://127.0.0.1:8000'

const activities = [
  'Walking',
  'Walking upstairs',
  'Walking downstairs',
  'Sitting',
  'Standing',
  'Laying',
]

const sensorData = Array.from({ length: 24 }, (_, i) => ({
  time: `${i * 5}s`,
  acceleration: Number(
    (0.45 + Math.sin(i * 0.8) * 0.28 + Math.cos(i * 0.35) * 0.12).toFixed(2),
  ),
  gyroscope: Number(
    (Math.sin(i * 0.65) * 0.35 + Math.cos(i * 0.3) * 0.15).toFixed(2),
  ),
}))

type Prediction = {
  activity: string
  confidence: number
  probabilities: Record<string, number>
}

function App() {
  const [apiOnline, setApiOnline] = useState(false)
  const [checking, setChecking] = useState(false)
  const [prediction, setPrediction] = useState<Prediction | null>(null)
  const [error, setError] = useState('')

  const [predicting, setPredicting] = useState(false)

  async function runPrediction() {
    setPredicting(true)
    setError('')

    try {
      const response = await fetch(`${API_URL}/predict/sample`, {
        method: 'POST',
      })

      if (!response.ok) {
        throw new Error('Prediction failed')
      }

      const result = await response.json()
      setPrediction(result)
    } catch {
      setError('Could not generate prediction. Check the backend endpoint.')
    } finally {
      setPredicting(false)
    }
  }

  async function checkBackend() {
    setChecking(true)
    try {
      const response = await fetch(`${API_URL}/health`)
      if (!response.ok) throw new Error('Backend health check failed')
      const data = await response.json()
      setApiOnline(Boolean(data))
      setError('')
    } catch {
      setApiOnline(false)
      setError('Cannot reach the API. Start your FastAPI backend on port 8000.')
    } finally {
      setChecking(false)
    }
  }

  useEffect(() => {
    void checkBackend()
  }, [])

  const confidence = prediction
    ? `${(prediction.confidence * 100).toFixed(1)}%`
    : '—'

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon"><Activity size={23} /></div>
          <div>
            <h1>ActivitySense</h1>
            <span>AI MONITORING</span>
          </div>
        </div>

        <div className="nav-label">WORKSPACE</div>
        <nav>
          <a className="nav-item active" href="#overview">
            <LayoutDashboard size={18} /> Overview
          </a>
          <a className="nav-item" href="#prediction">
            <BrainCircuit size={18} /> Activity prediction
          </a>
          <a className="nav-item" href="#sensors">
            <Waves size={18} /> Sensor analytics
          </a>
        </nav>

        <div className="sidebar-bottom">
          <div className="model-card">
            <div className="model-card-heading">
              <span className="model-dot" />
              <span>MODEL STATUS</span>
            </div>
            <h3>1D CNN</h3>
            <p>Human Activity Recognition</p>
            <div className="model-metric">
              <span>Test accuracy</span>
              <strong>92.87%</strong>
            </div>
            <div className="progress-track">
              <div className="progress-fill" style={{ width: '92.87%' }} />
            </div>
          </div>
          <div className="sidebar-footer">
            <ShieldCheck size={16} />
            <span>ActivitySense AI · v1.0</span>
          </div>
        </div>
      </aside>

      <main className="main-content" id="overview">
        <header className="topbar">
          <div>
            <div className="breadcrumb">Workspace / <strong>Overview</strong></div>
            <h2>Human Activity Dashboard</h2>
          </div>
          <div className="topbar-actions">
            <div className={`connection ${apiOnline ? 'online' : 'offline'}`}>
              <span className="status-dot" />
              {apiOnline ? 'API connected' : 'API disconnected'}
            </div>
            <button
              className="icon-button"
              onClick={() => void checkBackend()}
              title="Check backend connection"
              disabled={checking}
            >
              <RefreshCw size={17} className={checking ? 'spin' : ''} />
            </button>
            <div className="avatar">AS</div>
          </div>
        </header>

        {error && (
          <div className="error-banner">
            <CircleDot size={17} />
            <span>{error}</span>
            <button onClick={() => void checkBackend()}>Retry</button>
          </div>
        )}

        <section className="welcome-row">
          <div>
            <div className="eyebrow"><span /> MACHINE LEARNING · SENSOR INTELLIGENCE</div>
            <h1>Activity overview</h1>
            <p>Monitor human movement patterns through wearable sensor data.</p>
          </div>
          <div className="date-chip"><Clock3 size={16} /> HAR · 128 time steps</div>
        </section>

        <section className="stats-grid">
          <article className="stat-card">
            <div className="stat-top">
              <span>Latest prediction</span>
              <div className="stat-icon purple"><Activity size={19} /></div>
            </div>
            <h3 className="stat-value">{prediction?.activity ?? 'Awaiting data'}</h3>
            <p className="stat-foot">Recognized human activity</p>
          </article>

          <article className="stat-card">
            <div className="stat-top">
              <span>Prediction confidence</span>
              <div className="stat-icon green"><Gauge size={19} /></div>
            </div>
            <h3 className="stat-value">{confidence}</h3>
            <p className="stat-foot">Confidence of latest prediction</p>
          </article>

          <article className="stat-card">
            <div className="stat-top">
              <span>Activity classes</span>
              <div className="stat-icon blue"><HeartPulse size={19} /></div>
            </div>
            <h3 className="stat-value">06</h3>
            <p className="stat-foot">Supported recognition categories</p>
          </article>

          <article className="stat-card">
            <div className="stat-top">
              <span>Model accuracy</span>
              <div className="stat-icon orange"><BrainCircuit size={19} /></div>
            </div>
            <h3 className="stat-value">92.87%</h3>
            <p className="stat-foot">Measured on the official test set</p>
          </article>
        </section>

        <section className="content-grid">
          <article className="panel sensor-panel" id="sensors">
            <div className="panel-heading">
              <div>
                <h3>Sensor signal overview</h3>
                <p>Illustrative signal · not live device data</p>
              </div>
              <div className="sensor-badge"><Radio size={15} /> 9 channels</div>
            </div>
            <div className="chart-legend">
              <span><i className="legend-purple" /> Acceleration</span>
              <span><i className="legend-teal" /> Gyroscope</span>
            </div>
            <div className="chart-wrap">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={sensorData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid stroke="#edf0f5" strokeDasharray="4 4" vertical={false} />
                  <XAxis dataKey="time" tick={{ fill: '#8a91a2', fontSize: 11 }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fill: '#8a91a2', fontSize: 11 }} axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={{ borderRadius: 12, border: '1px solid #e9ecf3', fontSize: 12 }} />
                  <Line type="monotone" dataKey="acceleration" name="Acceleration" stroke="#7566e9" strokeWidth={2.5} dot={false} />
                  <Line type="monotone" dataKey="gyroscope" name="Gyroscope" stroke="#1bb8a5" strokeWidth={2.5} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div className="chart-note">
              <span className="note-indicator" />
              Sample visualization for dashboard layout
            </div>
          </article>

          <article className="panel prediction-panel" id="prediction">
            <div className="panel-heading">
              <div>
                <h3>Activity prediction</h3>
                <p>Model inference summary</p>
              </div>
              <div className="prediction-icon"><BrainCircuit size={20} /></div>
            </div>

            
            <div className="prediction-empty">
              <div className="empty-icon">
                <Activity size={27} />
              </div>

              <h4>{prediction ? prediction.activity : 'Ready for inference'}</h4>

              <p>
                {prediction
                  ? 'Prediction generated using a real sequence from the UCI HAR test dataset.'
                  : 'Run a sample through your trained CNN to see its prediction.'}
              </p>

              <button
                className="predict-button"
                onClick={runPrediction}
                disabled={predicting || !apiOnline}
              >
                {predicting ? 'Predicting...' : 'Run sample prediction'}
              </button>

              {prediction && (
                <div className="prediction-result">
                  <span>Model confidence</span>
                  <strong>{(prediction.confidence * 100).toFixed(2)}%</strong>
                </div>
              )}
            </div>

            <div className="prediction-details">
              <div><span>Model architecture</span><strong>1D CNN</strong></div>
              <div><span>Input shape</span><strong>128 × 9</strong></div>
              <div><span>API status</span><strong className={apiOnline ? 'text-green' : 'text-muted'}>{apiOnline ? 'Connected' : 'Offline'}</strong></div>
            </div>


            {prediction && (
              <div className="probability-list">
                <h4>Class probabilities</h4>

                {activities.map((activity) => {
                  const probability = prediction.probabilities[activity] ?? 0

                  return (
                    <div className="probability-item" key={activity}>
                      <div className="probability-label">
                        <span>{activity}</span>
                        <strong>{(probability * 100).toFixed(1)}%</strong>
                      </div>

                      <div className="progress-track">
                        <div
                          className="progress-fill"
                          style={{ width: `${probability * 100}%` }}
                        />
                      </div>
                    </div>
                  )
                })}
              </div>
            )}


            <div className="inference-hint">
              <Settings2 size={16} />
              <span>Live predictions will appear here once sensor input is connected.</span>
            </div>
          </article>
        </section>

        <section className="panel classes-panel">
          <div className="panel-heading">
            <div>
              <h3>Recognized activity classes</h3>
              <p>Categories supported by the trained model</p>
            </div>
            <span className="class-count">6 CLASSES</span>
          </div>
          <div className="class-grid">
            {activities.map((activity, index) => (
              <div className="class-item" key={activity}>
                <div className={`class-number color-${index + 1}`}>
                  {String(index + 1).padStart(2, '0')}
                </div>
                <span>{activity}</span>
                <CheckCircle2 size={16} className="class-check" />
              </div>
            ))}
          </div>
        </section>

        <footer className="page-footer">
          <span>ActivitySense AI</span>
          <span>Human Activity Recognition · UCI HAR Dataset</span>
        </footer>
      </main>
    </div>
  )
}

export default App

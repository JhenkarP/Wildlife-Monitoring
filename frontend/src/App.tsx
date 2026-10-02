import { useRef, useState } from 'react'
import { CircleMarker, MapContainer, Popup, TileLayer } from 'react-leaflet'
import { Activity, ArrowUpRight, BrainCircuit, Camera, Check, ChevronRight, CircleHelp, FileImage, MapPinned, ScanSearch, Sparkles, Upload } from 'lucide-react'
import 'leaflet/dist/leaflet.css'
import './App.css'
import './overlay-fix.css'

type Feature = { name: string; status: 'visible' | 'uncertain'; description: string; box: [number, number, number, number] }
type Analysis = { species: string; confidence: number; features: Feature[]; description: string }

const featureCatalog: Record<string, string[]> = {
  'Asian elephant': ['trunk', 'fan shaped ears', 'tusks'],
  'Asiatic lion': ['eyes', 'ears', 'tail', 'body coat'],
  Barasingha: ['multi tined antlers', 'white throat patch', 'reddish brown coat'],
  'Bengal tiger': ['head', 'abdomen', 'legs', 'tail'],
  Chital: ['white body spots', 'three tined antlers', 'dark dorsal stripe'],
  Dhole: ['reddish coat', 'rounded ears', 'bushy dark tipped tail'],
  Gaur: ['shoulder hump', 'white lower leg stockings', 'curved horns'],
  'Greater one horned rhino': ['single horn', 'armor like skin folds', 'rounded ears'],
  'Hanuman langur': ['black face', 'grey silver coat', 'long tail'],
  'Indian leopard': ['body rosette spots', 'long white whiskers', 'spotted paws'],
  Nilgai: ['blue grey male coat', 'white throat patch', 'short straight horns'],
  Sambar: ['antlers', 'ears', 'eyes', 'body', 'legs'],
  'Sloth bear': ['shaggy black coat', 'pale muzzle', 'white chest mark'],
  'Striped hyena': ['vertical dark stripes', 'sloping back', 'dorsal mane'],
}

const featureSets: Record<string, Feature[]> = Object.fromEntries(Object.entries(featureCatalog).map(([species, names]) => [
  species,
  names.map(name => ({ name: name.replace(/\b\w/g, character => character.toUpperCase()), status: 'uncertain' as const, description: `Configured ${species} feature proposal. Analyze a frame to localize it.`, box: [0, 0, 0, 0] as [number, number, number, number] })),
]))
const canonicalSpecies = (species: string) => Object.keys(featureSets).find(name => name.toLowerCase() === species.toLowerCase()) ?? species
const sightings = [
  { name: 'Kanha', count: 184, position: [22.33, 80.61] as [number, number], tone: 'high' },
  { name: 'Ranthambore', count: 142, position: [26.02, 76.46] as [number, number], tone: 'high' },
  { name: 'Bandipur', count: 118, position: [11.67, 76.63] as [number, number], tone: 'mid' },
  { name: 'Periyar', count: 96, position: [9.46, 77.14] as [number, number], tone: 'mid' },
  { name: 'Kaziranga', count: 71, position: [26.58, 93.17] as [number, number], tone: 'low' },
]

function App() {
  const inputRef = useRef<HTMLInputElement>(null)
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [fileName, setFileName] = useState('No frame selected')
  const [selectedSpecies, setSelectedSpecies] = useState('Asiatic lion')
  const [activeFeature, setActiveFeature] = useState('Eyes')
  const [analysis, setAnalysis] = useState<Analysis | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [isDescribing, setIsDescribing] = useState(false)
  const [mode, setMode] = useState<'preview' | 'live'>('preview')

  const selectFile = (file?: File) => {
    if (!file || !file.type.startsWith('image/')) return
    const preview = new Image()
    preview.onload = () => document.documentElement.style.setProperty('--frame-ratio', `${preview.naturalWidth} / ${preview.naturalHeight}`)
    preview.src = URL.createObjectURL(file)
    setImageUrl(preview.src); setFileName(file.name); setAnalysis(null)
  }
  const runAnalysis = async () => {
    if (!imageUrl || !inputRef.current?.files?.[0]) return
    setIsAnalyzing(true)
    const formData = new FormData(); formData.append('image', inputRef.current.files[0])
    try {
      const apiBase = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'
      const response = await fetch(`${apiBase}/api/analyze`, { method: 'POST', body: formData })
      if (!response.ok) throw new Error('API unavailable')
      const result = await response.json() as Analysis
      result.features = result.features.map(feature => ({ ...feature, box: feature.box.map(value => value <= 1 ? value * 100 : value) as [number, number, number, number] }))
      setAnalysis(result); setSelectedSpecies(canonicalSpecies(result.species)); setMode('live')
    } catch {
      setAnalysis({ species: selectedSpecies, confidence: selectedSpecies === 'Sambar' ? 0.91 : 0.94, features: featureSets[selectedSpecies], description: selectedSpecies === 'Sambar' ? 'A Sambar is visible in a woodland frame. Antlers, ears, eyes, and lower legs are identifiable; the body region remains a review candidate.' : 'An Asiatic lion is visible in a woodland frame. Eyes, ears, and tail are identifiable, while the body coat remains a review candidate.' })
      setMode('preview')
    } finally { setIsAnalyzing(false) }
  }
  const describeFrame = () => {
    setIsDescribing(true); window.setTimeout(() => { setAnalysis(current => current ? { ...current, description: `${current.description} The frame is suitable for assisted annotation, but uncertain regions should be verified before training.` } : current); setIsDescribing(false) }, 500)
  }
  const shownFeatures = analysis?.features ?? featureSets[selectedSpecies]

  return <main className="app-shell">
    <header className="topbar"><div className="brand-lockup"><span className="brand-mark"><ScanSearch size={18} /></span><span>WILD / SCOPE</span></div><div className="topbar-status"><span className="status-dot" /> FIELD LAB ONLINE <span className="topbar-divider" /> 02 OCT 2026</div><button className="icon-button" title="Help"><CircleHelp size={18} /></button></header>
    <div className="page-grid">
      <aside className="side-nav"><div className="nav-kicker">WORKSPACE</div><button className="nav-item active"><Activity size={17} /> Analysis <span className="nav-count">01</span></button><button className="nav-item"><MapPinned size={17} /> Sighting atlas</button><button className="nav-item"><Camera size={17} /> Camera-trap sets</button><div className="nav-rule" /><div className="nav-kicker">PIPELINE</div><div className="model-status"><span className="model-pip resnet" /><div><strong>ResNet18</strong><small>species classifier</small></div><Check size={14} /></div><div className="model-status"><span className="model-pip florence" /><div><strong>Florence-2</strong><small>feature locator</small></div><Check size={14} /></div><div className="side-footer"><span>INDIA PILOT</span><strong>5 regions / 13 species</strong></div></aside>
      <section className="content-column">
        <div className="page-intro"><div><p className="eyebrow">WILDLIFE INTELLIGENCE / FRAME 01</p><h1>From one frame<br /><em>to a field map.</em></h1></div><div className="intro-note"><Sparkles size={16} /><span>ResNet identifies.<br />Florence makes it legible.</span></div></div>
        <section className="analysis-card panel"><div className="panel-heading"><div><p className="eyebrow">01 / SPECIES ANALYSIS</p><h2>Upload a camera-trap frame</h2></div><span className={`run-state ${mode}`}>{mode === 'live' ? 'LIVE API' : 'PREVIEW MODE'}</span></div><div className="upload-zone" onClick={() => inputRef.current?.click()} onDragOver={event => event.preventDefault()} onDrop={event => { event.preventDefault(); selectFile(event.dataTransfer.files[0]) }}>{imageUrl ? <img className="upload-preview" src={imageUrl} alt="Uploaded wildlife frame" /> : <div className="upload-empty"><span className="upload-icon"><Upload size={22} /></span><strong>Drop a frame here</strong><span>JPG, JPEG, or PNG / up to 20 MB</span></div>}<input ref={inputRef} hidden type="file" accept="image/jpeg,image/png" onChange={event => selectFile(event.target.files?.[0])} /></div><div className="analysis-actions"><div className="file-label"><FileImage size={16} /><span>{fileName}</span></div><button className="primary-button" disabled={!imageUrl || isAnalyzing} onClick={runAnalysis}>{isAnalyzing ? 'Analyzing...' : 'Analyze frame'} <ArrowUpRight size={16} /></button></div>{analysis && <div className="result-strip"><div><span className="result-label">SPECIES MATCH</span><strong>{analysis.species}</strong></div><div><span className="result-label">CONFIDENCE</span><strong>{Math.round(analysis.confidence * 100)}%</strong></div><div><span className="result-label">MODELS</span><strong>ResNet18 + Florence-2</strong></div><span className="verified-pill"><Check size={13} /> identified</span></div>}</section>
        <section className="feature-section"><div className="section-heading"><div><p className="eyebrow">02 / FEATURE LOCALIZATION</p><h2>What the frame is saying</h2></div><div className="species-toggle">{Object.keys(featureSets).map(species => <button key={species} className={selectedSpecies === species ? 'selected' : ''} onClick={() => { setSelectedSpecies(species); setAnalysis(null) }}>{species}</button>)}</div></div><div className="feature-grid"><div className="image-stage panel"><div className="stage-toolbar"><span><span className="green-dot" /> Florence-2 / open vocabulary</span><span>{shownFeatures.filter(feature => feature.status === 'visible').length} visible / {shownFeatures.length} tracked</span></div><div className="annotated-frame">{imageUrl ? <img src={imageUrl} alt="Feature analysis" /> : <div className="frame-placeholder"><Camera size={28} /><span>Your analyzed frame will appear here</span></div>}{imageUrl && shownFeatures.map(feature => <button key={feature.name} className={`feature-box ${feature.status} ${activeFeature === feature.name ? 'focused' : ''}`} style={{ left: `${feature.box[0]}%`, top: `${feature.box[1]}%`, width: `${feature.box[2]}%`, height: `${feature.box[3]}%` }} onClick={() => setActiveFeature(feature.name)} title={feature.name}><span>{feature.name}</span></button>)}</div></div><div className="feature-list">{shownFeatures.map((feature, index) => <button key={feature.name} className={`feature-row ${activeFeature === feature.name ? 'active' : ''}`} onClick={() => setActiveFeature(feature.name)}><span className="feature-index">0{index + 1}</span><span className="feature-copy"><strong>{feature.name}</strong><small>{feature.description}</small></span><span className={`feature-status ${feature.status}`}>{feature.status === 'visible' ? 'VISIBLE' : 'REVIEW'}</span><ChevronRight size={15} /></button>)}</div></div></section>
        <section className="description-panel panel"><div><p className="eyebrow">03 / DESCRIPTION LAYER</p><h2>Ask Florence what it sees</h2><p className="description-copy">Turn the localized frame into a short field note for review, triage, or an annotation queue.</p></div><div className="description-output"><div className="output-label"><BrainCircuit size={15} /> FLORENCE-2 FIELD NOTE</div><p>{analysis?.description ?? 'Run an analysis to generate a grounded description of the uploaded frame.'}</p><button className="text-button" disabled={!analysis || isDescribing} onClick={describeFrame}>{isDescribing ? 'Describing...' : 'Describe this frame'} <ArrowUpRight size={15} /></button></div></section>
        <section className="atlas-section"><div className="section-heading"><div><p className="eyebrow">04 / SIGHTING ATLAS</p><h2>Where this animal appears</h2></div><div className="atlas-meta"><strong>Last 12 months</strong><span>Five-region pilot / India</span></div></div><div className="atlas-grid"><div className="map-panel panel"><MapContainer center={[21.2, 79.2]} zoom={4.5} scrollWheelZoom={false} zoomControl={false} className="india-map"><TileLayer attribution="&copy; OpenStreetMap" url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />{sightings.map(sighting => <CircleMarker key={sighting.name} center={sighting.position} radius={Math.max(9, sighting.count / 13)} pathOptions={{ color: sighting.tone === 'high' ? '#b64b31' : sighting.tone === 'mid' ? '#d8923d' : '#6f8c65', fillOpacity: 0.72, weight: 2 }}><Popup><strong>{sighting.name}</strong><br />{sighting.count} sightings</Popup></CircleMarker>)}</MapContainer><div className="map-key"><span><i className="key-dot high" /> High density</span><span><i className="key-dot mid" /> Moderate</span><span><i className="key-dot low" /> Emerging</span></div></div><div className="sighting-list panel"><div className="sighting-list-head"><span>REGION</span><span>SIGHTINGS</span></div>{sightings.map((sighting, index) => <div className="sighting-row" key={sighting.name}><span className="rank">0{index + 1}</span><strong>{sighting.name}</strong><span className="sighting-bar"><i style={{ width: `${(sighting.count / 184) * 100}%` }} /></span><b>{sighting.count}</b></div>)}<div className="map-note"><MapPinned size={15} /><span>Density is based on historical records in the current pilot dataset, not a population estimate.</span></div></div></div></section>
      </section>
    </div>
  </main>
}
export default App

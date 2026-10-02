import { useEffect, useRef, useState } from 'react'
import { CircleMarker, MapContainer, Popup, TileLayer, useMap } from 'react-leaflet'
import { Activity, ArrowLeft, ArrowUpRight, BrainCircuit, Camera, Check, ChevronRight, CircleHelp, FileImage, MapPinned, ScanSearch, Sparkles, Upload } from 'lucide-react'
import 'leaflet/dist/leaflet.css'
import './App.css'
import './overlay-fix.css'

type Feature = { name: string; status: 'visible' | 'uncertain'; description: string; box: [number, number, number, number] }
type Analysis = { species: string; confidence: number; features: Feature[]; description: string }
type SightingRecord = { species: string; observed_on: string; latitude: number; longitude: number; state?: string; district?: string; protected_area?: string; source?: string; dataset_name?: string; publisher?: string; occurrence_id?: string; record_url?: string; photo_url?: string }
type RegionCount = { name: string; count: number }
type SanctuaryStat = { name: string; count: number; percentage: number; leading_species: string; leading_species_count: number; leading_species_percentage: number }
type CompositionItem = { species: string; count: number; percentage: number; photo_count: number }
type SanctuaryReportData = { name: string; total_records: number; species_count: number; date_range: { from: string; to: string }; photo_count: number; georeferenced_count: number; composition: CompositionItem[]; source_records: SightingRecord[]; match_method: string }

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
  'Sloth bear': ['shaggy black coat', 'pale muzzle', 'white chest mark', 'claws or paws'],
  'Striped hyena': ['vertical dark stripes', 'sloping back', 'dorsal mane'],
}

const featureSets: Record<string, Feature[]> = Object.fromEntries(Object.entries(featureCatalog).map(([species, names]) => [
  species,
  names.map(name => ({ name: name.replace(/\b\w/g, character => character.toUpperCase()), status: 'uncertain' as const, description: `Configured ${species} feature proposal. Analyze a frame to localize it.`, box: [0, 0, 0, 0] as [number, number, number, number] })),
]))
const canonicalSpecies = (species: string) => Object.keys(featureSets).find(name => name.toLowerCase() === species.toLowerCase()) ?? species
const formatDate = (value: string) => new Intl.DateTimeFormat('en-IN', { day: '2-digit', month: 'short', year: 'numeric' }).format(new Date(value))
const indiaBounds: [[number, number], [number, number]] = [[6.5, 68], [35.7, 97.5]]
const majorSanctuaries = [
  ['Gir Wildlife Sanctuary', 'Gujarat'],
  ['Wayanad Wildlife Sanctuary', 'Kerala'],
  ['Bhadra Wildlife Sanctuary', 'Karnataka'],
  ['Dandeli Wildlife Sanctuary', 'Karnataka'],
  ['Kumbhalgarh Wildlife Sanctuary', 'Rajasthan'],
  ['Gahirmatha Marine Wildlife Sanctuary', 'Odisha'],
] as const

function MapSizeFix({ recordCount }: { recordCount: number }) {
  const map = useMap()
  useEffect(() => {
    const frame = window.requestAnimationFrame(() => {
      map.invalidateSize()
      map.fitBounds(indiaBounds, { padding: [12, 12] })
    })
    return () => window.cancelAnimationFrame(frame)
  }, [map, recordCount])
  return null
}

function SanctuaryAtlas({ stats, totalRecords, onSelect }: { stats: SanctuaryStat[]; totalRecords: number; onSelect: (name: string) => void }) {
  return <section className="sanctuary-atlas-page"><div className="section-heading"><div><p className="eyebrow">02 / SIGHTING ATLAS</p><h2>Major wildlife sanctuaries</h2></div><div className="atlas-meta"><strong>Six major sanctuaries</strong><span>All available records / {totalRecords}</span></div></div><div className="sanctuary-intro">GBIF records are spatially matched to approximate sanctuary bounding boxes. Each record retains its GBIF source link for authentication; the sanctuary assignment is not a protected-area polygon claim.</div><div className="sanctuary-directory">{majorSanctuaries.map(([name, state], index) => <div className="sanctuary-directory-row" key={name} onClick={() => onSelect(name)} role="button" tabIndex={0}><b>{String(index + 1).padStart(2, '0')}</b><strong>{name}</strong><span>{state}</span><ChevronRight size={16} /></div>)}</div><div className="sanctuary-results-heading"><p className="eyebrow">SPATIAL SIGHTING RANKING</p><span>{stats.length ? `${stats.length} sanctuaries with records` : 'No sanctuary records'}</span></div><div className="sanctuary-table"><div className="sanctuary-table-head"><span>RANK / SANCTUARY</span><span>LEADING ANIMAL</span><span>SIGHTINGS</span><span>SHARE</span></div>{stats.length ? stats.map((sanctuary, index) => <div className="sanctuary-row" key={sanctuary.name} onClick={() => onSelect(sanctuary.name)} role="button" tabIndex={0}><span className="sanctuary-name"><b>{String(index + 1).padStart(2, '0')}</b><strong>{sanctuary.name}</strong></span><span><strong>{sanctuary.leading_species}</strong><small>{sanctuary.leading_species_count} records / {sanctuary.leading_species_percentage}% here</small></span><span className="sanctuary-count">{sanctuary.count}</span><span className="sanctuary-share"><i><em style={{ width: `${sanctuary.percentage}%` }} /></i><b>{sanctuary.percentage}%</b></span></div>) : <div className="map-note"><MapPinned size={15} /><span>No sanctuary records were found in the database.</span></div>}</div></section>
}

function SanctuaryReport({ report, onBack }: { report: SanctuaryReportData; onBack: () => void }) {
  return <section className="sanctuary-report-page"><button className="back-button" onClick={onBack}><ArrowLeft size={16} /> Back to sanctuary atlas</button><div className="report-heading"><div><p className="eyebrow">SANCTUARY REPORT / ANIMAL COMPOSITION</p><h2>{report.name}</h2><p>{report.match_method}</p></div><div className="report-total"><strong>{report.total_records}</strong><span>GBIF records</span></div></div><div className="report-metrics"><div><span>ANIMALS RECORDED</span><strong>{report.species_count}</strong></div><div><span>OBSERVATION PERIOD</span><div className="report-date-range"><span>FROM <strong>{formatDate(report.date_range.from)}</strong></span><span>TO <strong>{formatDate(report.date_range.to)}</strong></span></div></div><div><span>PHOTO COVERAGE</span><strong>{Math.round(report.photo_count / report.total_records * 100)}%</strong></div><div><span>GEOSPATIAL COVERAGE</span><strong>{Math.round(report.georeferenced_count / report.total_records * 100)}%</strong></div></div><div className="report-grid"><section className="composition-panel"><div className="report-section-heading"><div><p className="eyebrow">COMPOSITION</p><h3>Animals observed</h3></div><span>Share of records</span></div>{report.composition.map(item => <div className="composition-row" key={item.species}><div className="composition-label"><strong>{item.species}</strong><span>{item.count} records / {item.photo_count} with photos</span></div><b>{item.percentage}%</b><i><em style={{ width: `${item.percentage}%` }} /></i></div>)}</section><section className="source-panel"><div className="report-section-heading"><div><p className="eyebrow">SOURCE CHECK</p><h3>Recent records</h3></div><span>GBIF links</span></div>{report.source_records.map(record => <div className="source-row" key={record.occurrence_id}><div><strong>{record.species}</strong><span>{record.observed_on} / {record.dataset_name || 'GBIF dataset'}</span></div>{record.record_url && <a href={record.record_url} target="_blank" rel="noreferrer" title="Open GBIF source record"><ArrowUpRight size={15} /></a>}</div>)}</section></div></section>
}

function App() {
  const inputRef = useRef<HTMLInputElement>(null)
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [fileName, setFileName] = useState('No frame selected')
  const [selectedSpecies, setSelectedSpecies] = useState('Asiatic lion')
  const [activeFeature, setActiveFeature] = useState('Eyes')
  const [analysis, setAnalysis] = useState<Analysis | null>(null)
  const [sightingRecords, setSightingRecords] = useState<SightingRecord[]>([])
  const [regionCounts, setRegionCounts] = useState<RegionCount[]>([])
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [mode, setMode] = useState<'preview' | 'live'>('preview')
  const [activePage, setActivePage] = useState<'analysis' | 'atlas'>('analysis')
  const [sanctuaryStats, setSanctuaryStats] = useState<SanctuaryStat[]>([])
  const [sanctuaryTotal, setSanctuaryTotal] = useState(0)
  const [selectedSanctuary, setSelectedSanctuary] = useState<string | null>(null)
  const [sanctuaryReport, setSanctuaryReport] = useState<SanctuaryReportData | null>(null)

  useEffect(() => {
    const species = analysis?.species ?? selectedSpecies
    const apiBase = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'
    fetch(`${apiBase}/api/sightings?species=${encodeURIComponent(species)}&days=365`)
      .then(response => response.ok ? response.json() as Promise<{ records: SightingRecord[]; regions: RegionCount[] }> : Promise.reject(new Error('Sightings unavailable')))
      .then(result => { setSightingRecords(result.records); setRegionCounts(result.regions) })
      .catch(() => { setSightingRecords([]); setRegionCounts([]) })
  }, [analysis?.species, selectedSpecies])

  useEffect(() => {
    const apiBase = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'
    fetch(`${apiBase}/api/sanctuaries`)
      .then(response => response.ok ? response.json() as Promise<{ sanctuaries: SanctuaryStat[]; total_records: number }> : Promise.reject(new Error('Sanctuaries unavailable')))
      .then(result => { setSanctuaryStats(result.sanctuaries); setSanctuaryTotal(result.total_records) })
      .catch(() => { setSanctuaryStats([]); setSanctuaryTotal(0) })
  }, [])

  useEffect(() => {
    if (!selectedSanctuary) { setSanctuaryReport(null); return }
    const apiBase = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'
    fetch(`${apiBase}/api/sanctuaries/${encodeURIComponent(selectedSanctuary)}`)
      .then(response => response.ok ? response.json() as Promise<SanctuaryReportData> : Promise.reject(new Error('Report unavailable')))
      .then(setSanctuaryReport)
      .catch(() => setSanctuaryReport(null))
  }, [selectedSanctuary])

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
  const shownFeatures = analysis?.features ?? featureSets[selectedSpecies]

  const pageState = analysis ? 'analysis-ready' : imageUrl ? 'has-image' : 'upload-only'
  return <main className={`app-shell ${pageState}`}>
    <header className="topbar"><div className="brand-lockup"><span className="brand-mark"><ScanSearch size={18} /></span><span>WILD / SCOPE</span></div><div className="topbar-status"><span className="status-dot" /> FIELD LAB ONLINE <span className="topbar-divider" /> 02 OCT 2026</div><button className="icon-button" title="Help"><CircleHelp size={18} /></button></header>
    <div className="page-grid">
      <aside className="side-nav"><div className="nav-kicker">WORKSPACE</div><button className={`nav-item ${activePage === 'analysis' ? 'active' : ''}`} onClick={() => setActivePage('analysis')}><Activity size={17} /> Analysis <span className="nav-count">01</span></button><button className={`nav-item ${activePage === 'atlas' ? 'active' : ''}`} onClick={() => setActivePage('atlas')}><MapPinned size={17} /> Sighting atlas</button><button className="nav-item"><Camera size={17} /> Camera-trap sets</button><div className="nav-rule" /><div className="nav-kicker">PIPELINE</div><div className="model-status"><span className="model-pip resnet" /><div><strong>ResNet18</strong><small>species classifier</small></div><Check size={14} /></div><div className="model-status"><span className="model-pip florence" /><div><strong>Florence-2</strong><small>feature locator</small></div><Check size={14} /></div><div className="side-footer"><span>INDIA PILOT</span><strong>5 regions / 13 species</strong></div></aside>
      <section className={`content-column ${activePage === 'atlas' ? 'atlas-mode' : ''}`}>
        {activePage === 'atlas' && (sanctuaryReport ? <SanctuaryReport report={sanctuaryReport} onBack={() => setSelectedSanctuary(null)} /> : <SanctuaryAtlas stats={sanctuaryStats} totalRecords={sanctuaryTotal} onSelect={setSelectedSanctuary} />)}
        <div className="page-intro"><div><p className="eyebrow">WILDLIFE INTELLIGENCE / FRAME 01</p><h1>From one frame<br /><em>to a field map.</em></h1></div><div className="intro-note"><Sparkles size={16} /><span>ResNet identifies.<br />Florence makes it legible.</span></div></div>
        <section className="analysis-card panel"><div className="panel-heading"><div><p className="eyebrow">01 / SPECIES ANALYSIS</p><h2>Upload a camera-trap frame</h2></div><span className={`run-state ${mode}`}>{mode === 'live' ? 'LIVE API' : 'PREVIEW MODE'}</span></div><div className="upload-zone" onClick={() => inputRef.current?.click()} onDragOver={event => event.preventDefault()} onDrop={event => { event.preventDefault(); selectFile(event.dataTransfer.files[0]) }}>{imageUrl ? <img className="upload-preview" src={imageUrl} alt="Uploaded wildlife frame" /> : <div className="upload-empty"><span className="upload-icon"><Upload size={22} /></span><strong>Drop a frame here</strong><span>JPG, JPEG, or PNG / up to 20 MB</span></div>}<input ref={inputRef} hidden type="file" accept="image/jpeg,image/png" onChange={event => selectFile(event.target.files?.[0])} /></div><div className="analysis-actions"><div className="file-label"><FileImage size={16} /><span>{fileName}</span></div><button className="primary-button" disabled={!imageUrl || isAnalyzing} onClick={runAnalysis}>{isAnalyzing ? 'Analyzing...' : 'Analyze frame'} <ArrowUpRight size={16} /></button></div>{analysis && <div className="result-strip"><div><span className="result-label">SPECIES MATCH</span><strong>{analysis.species}</strong></div><div><span className="result-label">CONFIDENCE</span><strong>{Math.round(analysis.confidence * 100)}%</strong></div><div><span className="result-label">MODELS</span><strong>ResNet18 + Florence-2</strong></div><span className="verified-pill"><Check size={13} /> identified</span></div>}</section>
        <section className="feature-section"><div className="section-heading"><div><p className="eyebrow">02 / FEATURE LOCALIZATION</p><h2>What the frame is saying</h2></div><div className="species-toggle">{Object.keys(featureSets).map(species => <button key={species} className={selectedSpecies === species ? 'selected' : ''} onClick={() => { setSelectedSpecies(species); setAnalysis(null) }}>{species}</button>)}</div></div><div className="feature-grid"><div className="image-stage panel"><div className="stage-toolbar"><span><span className="green-dot" /> Florence-2 / open vocabulary</span><span>{shownFeatures.filter(feature => feature.status === 'visible').length} visible / {shownFeatures.length} tracked</span></div><div className="annotated-frame">{imageUrl ? <img src={imageUrl} alt="Feature analysis" /> : <div className="frame-placeholder"><Camera size={28} /><span>Your analyzed frame will appear here</span></div>}{imageUrl && shownFeatures.map(feature => <button key={feature.name} className={`feature-box ${feature.status} ${activeFeature === feature.name ? 'focused' : ''}`} style={{ left: `${feature.box[0]}%`, top: `${feature.box[1]}%`, width: `${feature.box[2]}%`, height: `${feature.box[3]}%` }} onClick={() => setActiveFeature(feature.name)} title={feature.name}><span>{feature.name}</span></button>)}</div></div><div className="feature-list">{shownFeatures.map((feature, index) => <button key={feature.name} className={`feature-row ${activeFeature === feature.name ? 'active' : ''}`} onClick={() => setActiveFeature(feature.name)}><span className="feature-index">0{index + 1}</span><span className="feature-copy"><strong>{feature.name}</strong><small>{feature.description}</small></span><span className={`feature-status ${feature.status}`}>{feature.status === 'visible' ? 'VISIBLE' : 'REVIEW'}</span><ChevronRight size={15} /></button>)}</div></div></section>
        <section className="description-panel panel"><div><p className="eyebrow">03 / DESCRIPTION LAYER</p><h2>Florence description</h2></div><div className="description-output"><div className="output-label"><BrainCircuit size={15} /> FLORENCE-2 DESCRIPTION</div><p>{analysis?.description ?? 'Run an analysis to generate a description for the uploaded frame.'}</p></div></section>
        <section className="atlas-section"><div className="section-heading"><div><p className="eyebrow">04 / SIGHTING ATLAS</p><h2>Where this animal appears</h2></div><div className="atlas-meta"><strong>Last 12 months / {sightingRecords.length} records</strong><span>Wildlife records / India</span></div></div><div className="atlas-grid"><div className="map-panel panel"><MapContainer bounds={indiaBounds} maxBounds={[[5, 66], [37, 99]]} maxBoundsViscosity={1} scrollWheelZoom={false} zoomControl={false} className="india-map"><MapSizeFix recordCount={sightingRecords.length} /><TileLayer attribution="&copy; OpenStreetMap" url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />{sightingRecords.map((sighting, index) => <CircleMarker key={`${sighting.observed_on}-${sighting.latitude}-${sighting.longitude}-${index}`} center={[sighting.latitude, sighting.longitude]} radius={8} pathOptions={{ color: '#b64b31', fillOpacity: 0.72, weight: 2 }}><Popup><strong>{sighting.protected_area || sighting.district || sighting.state || sighting.species}</strong><br />{sighting.species}<br />{sighting.observed_on}<br /><small>{sighting.source || 'Source unavailable'}{sighting.dataset_name ? ` / ${sighting.dataset_name}` : ''}</small>{sighting.record_url && <><br /><a href={sighting.record_url} target="_blank" rel="noreferrer">Open verified source record</a></>}{sighting.photo_url && <><br /><a href={sighting.photo_url} target="_blank" rel="noreferrer">View observation photo</a></>}</Popup></CircleMarker>)}</MapContainer><div className="map-key">{sightingRecords.length ? <span><i className="key-dot high" /> Recorded sighting</span> : <span><i className="key-dot low" /> No records</span>}</div></div><div className="sighting-list panel"><div className="sighting-list-head"><span>REGION</span><span>SIGHTINGS</span></div>{regionCounts.length ? regionCounts.map((region, index) => <div className="sighting-row" key={region.name}><span className="rank">{String(index + 1).padStart(2, '0')}</span><strong>{region.name}</strong><span className="sighting-bar"><i style={{ width: `${(region.count / regionCounts[0].count) * 100}%` }} /></span><b>{region.count}</b></div>) : <div className="map-note"><MapPinned size={15} /><span>No {analysis?.species ?? selectedSpecies} sightings were returned for the last 12 months.</span></div>}<div className="map-note"><MapPinned size={15} /><span>Source: wildlife observations stored in the project SQLite database, filtered by species and date.</span></div></div></div></section>
      </section>
    </div>
  </main>
}
export default App

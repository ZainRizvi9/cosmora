import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { Stars } from '@react-three/drei'
import { useRef, useState, useEffect, useMemo } from 'react'
import * as THREE from 'three'

const KEPLER = {
  Mercury: { a:0.38709927, e:0.20563593, L:252.25032350, w:77.45779628  },
  Venus:   { a:0.72333566, e:0.00677672, L:181.97909950, w:131.60246718 },
  Earth:   { a:1.00000261, e:0.01671123, L:100.46457166, w:102.93768193 },
  Mars:    { a:1.52371034, e:0.09339410, L:-4.55343205,  w:-23.94362959 },
  Jupiter: { a:5.20288700, e:0.04838624, L:34.39644051,  w:14.72847983  },
  Saturn:  { a:9.53667594, e:0.05386179, L:49.95424423,  w:92.59887831  },
  Uranus:  { a:19.18916464,e:0.04725744, L:313.23810451, w:170.95427630 },
  Neptune: { a:30.06992276,e:0.00859048, L:-55.12002969, w:44.96476227  },
}

const AU = 10
const NASA_KEY = import.meta.env.VITE_NASA_KEY

function keplerPos(name, jd) {
  const T = (jd - 2451545.0) / 36525
  const el = KEPLER[name]
  let M = ((el.L - el.w) * Math.PI / 180) + T * 0.01
  let E = M
  for (let i = 0; i < 10; i++) E = M + el.e * Math.sin(E)
  const x = el.a * (Math.cos(E) - el.e)
  const y = el.a * Math.sqrt(1 - el.e * el.e) * Math.sin(E)
  const v = Math.atan2(y, x)
  const dist = Math.sqrt(x*x + y*y)
  return new THREE.Vector3(dist * Math.cos(v) * AU, 0, dist * Math.sin(v) * AU)
}

function dateToJD(date) {
  return new Date(date).getTime() / 86400000 + 2440587.5
}

function detectAlignments(positions) {
  const names = Object.keys(positions)
  const angles = {}
  names.forEach(n => {
    angles[n] = Math.atan2(positions[n].z, positions[n].x) * 180 / Math.PI
  })
  const alignments = []
  for (let i = 0; i < names.length; i++) {
    for (let j = i+1; j < names.length; j++) {
      for (let k = j+1; k < names.length; k++) {
        const a = angles[names[i]], b = angles[names[j]], c = angles[names[k]]
        const spread = Math.max(a,b,c) - Math.min(a,b,c)
        if (Math.min(spread, 360 - spread) < 15)
          alignments.push([names[i], names[j], names[k]])
      }
    }
  }
  return alignments
}

const KNN_TRAINING = [
  { size:0.05, vel:20000, dist:500000, label:'low'     },
  { size:0.1,  vel:30000, dist:400000, label:'low'     },
  { size:0.05, vel:15000, dist:700000, label:'low'     },
  { size:0.3,  vel:50000, dist:200000, label:'medium'  },
  { size:0.4,  vel:60000, dist:150000, label:'medium'  },
  { size:0.2,  vel:45000, dist:250000, label:'medium'  },
  { size:0.8,  vel:70000, dist:80000,  label:'high'    },
  { size:1.2,  vel:80000, dist:50000,  label:'high'    },
  { size:0.6,  vel:75000, dist:60000,  label:'high'    },
  { size:2.0,  vel:90000, dist:20000,  label:'extreme' },
  { size:1.5,  vel:85000, dist:30000,  label:'extreme' },
  { size:0.02, vel:10000, dist:900000, label:'low'     },
  { size:0.35, vel:55000, dist:180000, label:'medium'  },
  { size:0.9,  vel:72000, dist:70000,  label:'high'    },
]

function normalize(val, min, max) {
  return (val - min) / (max - min)
}

function knnClassify(size, vel, dist, k = 3) {
  const sn = normalize(size, 0, 2)
  const vn = normalize(vel, 0, 90000)
  const dn = normalize(dist, 0, 900000)
  const sorted = KNN_TRAINING.map(t => ({
    label: t.label,
    d: Math.sqrt(
      (normalize(t.size, 0, 2)      - sn) ** 2 +
      (normalize(t.vel,  0, 90000)  - vn) ** 2 +
      (normalize(t.dist, 0, 900000) - dn) ** 2
    )
  })).sort((a, b) => a.d - b.d).slice(0, k)
  const votes = {}
  sorted.forEach(t => { votes[t.label] = (votes[t.label] || 0) + 1 })
  return Object.entries(votes).sort((a, b) => b[1] - a[1])[0][0]
}

const THREAT_COLORS = {
  low: '#44ff88',
  medium: '#ffcc00',
  high: '#ff8800',
  extreme: '#ff2200'
}

const SPACE_HISTORY = [
  { month:1,  day:27, text:'Apollo 1 fire kills three astronauts (1967)' },
  { month:1,  day:28, text:'Space Shuttle Challenger disaster (1986)' },
  { month:2,  day:1,  text:'Space Shuttle Columbia disintegrates on re-entry (2003)' },
  { month:2,  day:18, text:'Perseverance rover lands on Mars (2021)' },
  { month:3,  day:18, text:'Voskhod 2: first spacewalk by Alexei Leonov (1965)' },
  { month:4,  day:12, text:'Yuri Gagarin becomes first human in space (1961)' },
  { month:4,  day:12, text:'First Space Shuttle Columbia launch (1981)' },
  { month:4,  day:24, text:'Hubble Space Telescope launched (1990)' },
  { month:5,  day:5,  text:'Alan Shepard becomes first American in space (1961)' },
  { month:5,  day:25, text:'JFK challenges the US to land on the Moon (1961)' },
  { month:6,  day:16, text:'Valentina Tereshkova becomes first woman in space (1963)' },
  { month:7,  day:4,  text:'Mars Pathfinder lands on Mars (1997)' },
  { month:7,  day:11, text:'Skylab re-enters atmosphere (1979)' },
  { month:7,  day:20, text:'Apollo 11: humans walk on the Moon for the first time (1969)' },
  { month:8,  day:20, text:'Voyager 2 launched toward the outer solar system (1977)' },
  { month:9,  day:5,  text:'Voyager 1 launched — now the farthest human-made object (1977)' },
  { month:10, day:4,  text:'Sputnik 1 launched — first artificial satellite (1957)' },
  { month:10, day:15, text:'Cassini spacecraft enters Saturn orbit (1997)' },
  { month:11, day:2,  text:'First crew arrives at the International Space Station (2000)' },
  { month:11, day:19, text:'Apollo 12 lands on the Moon (1969)' },
  { month:12, day:7,  text:'Galileo spacecraft arrives at Jupiter (1995)' },
  { month:12, day:19, text:'Hubble Space Telescope repaired by astronauts (1993)' },
  { month:12, day:24, text:'Apollo 8 orbits the Moon on Christmas Eve (1968)' },
]

function getHistoryForDate(dateStr) {
  const [, m, d] = dateStr.split('-').map(Number)
  return SPACE_HISTORY.filter(e => e.month === m && e.day === d)
}

const PLANET_DATA = [
  { name:'Mercury', texture:'/textures/2k_mercury.jpg',       size:0.38, tilt:0.03, color:'#b5b5b5', distAU:'0.39 AU', moons:0,   info:'Closest planet to the Sun. Temperatures swing from -180°C to 430°C with no atmosphere to regulate them.' },
  { name:'Venus',   texture:'/textures/2k_venus_surface.jpg', size:0.95, tilt:0.05, color:'#e8cda0', distAU:'0.72 AU', moons:0,   info:'Hottest planet at 465°C despite not being closest to the Sun. Thick CO₂ atmosphere. Spins backwards.' },
  { name:'Earth',   texture:'/textures/2k_earth_daymap.jpg',  size:1.0,  tilt:0.41, color:'#2f6aff', distAU:'1.00 AU', moons:1,   info:'Our home. The only planet known to harbour life. 71% of the surface is covered in water.' },
  { name:'Mars',    texture:'/textures/2k_mars.jpg',          size:0.53, tilt:0.44, color:'#c1440e', distAU:'1.52 AU', moons:2,   info:'The Red Planet. Home to Olympus Mons — the tallest volcano in the solar system at 22km.' },
  { name:'Jupiter', texture:'/textures/2k_jupiter.jpg',       size:2.0,  tilt:0.05, color:'#c88b3a', distAU:'5.20 AU', moons:95,  info:'The largest planet — over 1,300 Earths could fit inside. The Great Red Spot has been raging for 350+ years.' },
  { name:'Saturn',  texture:'/textures/2k_saturn.jpg',        size:1.7,  tilt:0.47, color:'#e4d191', distAU:'9.58 AU', moons:146, rings:true, info:'Known for its stunning ring system made of ice and rock. So low in density it would float on water.' },
  { name:'Uranus',  texture:'/textures/2k_uranus.jpg',        size:1.3,  tilt:1.71, color:'#7de8e8', distAU:'19.2 AU', moons:28,  info:'Rotates on its side at 98°. Has the coldest atmosphere of any planet at -224°C.' },
  { name:'Neptune', texture:'/textures/2k_neptune.jpg',       size:1.2,  tilt:0.49, color:'#3f54ba', distAU:'30.1 AU', moons:16,  info:'The windiest planet — gusts reach 2,100 km/h. Takes 165 Earth years to complete one orbit.' },
]

const loader = new THREE.TextureLoader()
const texCache = {}
PLANET_DATA.forEach(p => { texCache[p.name] = loader.load(p.texture) })
const sunTex = loader.load('/textures/2k_sun.jpg')
const ringTex = loader.load('/textures/2k_saturn_ring_alpha.png')

function Sun({ onClick }) {
  const ref = useRef()
  useFrame(({ clock }) => { ref.current.rotation.y = clock.getElapsedTime() * 0.05 })
  return (
    <mesh ref={ref} onClick={onClick}
      onPointerOver={() => document.body.style.cursor = 'pointer'}
      onPointerOut={() => document.body.style.cursor = 'default'}
    >
      <sphereGeometry args={[2.5, 64, 64]} />
      <meshStandardMaterial map={sunTex} emissive="#ff4400" emissiveIntensity={0.8} />
    </mesh>
  )
}

function OrbitRing({ position }) {
  const dist = Math.sqrt(position.x ** 2 + position.z ** 2)
  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]}>
      <ringGeometry args={[dist - 0.02, dist + 0.02, 128]} />
      <meshBasicMaterial color="#ffffff" opacity={0.07} transparent side={THREE.DoubleSide} />
    </mesh>
  )
}

function Planet({ data, position, onSelect, isTarget }) {
  const meshRef = useRef()
  const [hovered, setHovered] = useState(false)
  useFrame(() => { if (meshRef.current) meshRef.current.rotation.y += 0.004 })
  return (
    <group position={[position.x, position.y, position.z]}>
      <mesh
        ref={meshRef}
        rotation={[data.tilt, 0, 0]}
        scale={hovered || isTarget ? 1.15 : 1}
        onClick={() => onSelect(data)}
        onPointerOver={() => { setHovered(true); document.body.style.cursor = 'pointer' }}
        onPointerOut={() => { setHovered(false); document.body.style.cursor = 'default' }}
      >
        <sphereGeometry args={[data.size, 64, 64]} />
        <meshStandardMaterial map={texCache[data.name]} />
      </mesh>
      {data.rings && (
        <mesh rotation={[Math.PI / 2 - 0.4, 0, 0]}>
          <ringGeometry args={[data.size * 1.35, data.size * 2.2, 128]} />
          <meshBasicMaterial map={ringTex} side={THREE.DoubleSide} transparent opacity={0.85} />
        </mesh>
      )}
      {isTarget && (
        <mesh rotation={[-Math.PI / 2, 0, 0]}>
          <ringGeometry args={[data.size * 1.8, data.size * 1.95, 64]} />
          <meshBasicMaterial color="#00ffff" transparent opacity={0.6} side={THREE.DoubleSide} />
        </mesh>
      )}
    </group>
  )
}

function ISS({ onSelect }) {
  const ref = useRef()
  const [pos, setPos] = useState(new THREE.Vector3(12, 0.8, 0))
  const [issData, setIssData] = useState(null)

  useEffect(() => {
    const fetchISS = () => {
      fetch('https://api.wheretheiss.at/v1/satellites/25544')
        .then(r => r.json())
        .then(d => {
          setIssData(d)
          const angle = (d.longitude / 180) * Math.PI
          setPos(new THREE.Vector3(Math.cos(angle) * 11.5, 0.8, Math.sin(angle) * 11.5))
        }).catch(() => {})
    }
    fetchISS()
    const id = setInterval(fetchISS, 5000)
    return () => clearInterval(id)
  }, [])

  useFrame(() => { if (ref.current) ref.current.rotation.y += 0.02 })

  return (
    <group position={[pos.x, pos.y, pos.z]}>
      <mesh
        ref={ref}
        onClick={() => onSelect({
          name: 'ISS — International Space Station',
          info: `Altitude: ${issData?.altitude?.toFixed(1) ?? '~408'} km · Speed: ${issData?.velocity?.toFixed(0) ?? '~27,600'} km/h · Position: ${issData ? `${issData.latitude?.toFixed(2)}°, ${issData.longitude?.toFixed(2)}°` : 'loading...'}`
        })}
        onPointerOver={() => document.body.style.cursor = 'pointer'}
        onPointerOut={() => document.body.style.cursor = 'default'}
      >
        <boxGeometry args={[0.25, 0.06, 0.5]} />
        <meshStandardMaterial color="#ccccff" metalness={0.8} roughness={0.2} />
      </mesh>
      <mesh>
        <boxGeometry args={[0.8, 0.02, 0.06]} />
        <meshStandardMaterial color="#4488ff" metalness={0.5} roughness={0.3} />
      </mesh>
    </group>
  )
}

function Asteroid({ position, data, onSelect }) {
  const ref = useRef()
  useFrame(() => {
    ref.current.rotation.x += 0.01
    ref.current.rotation.y += 0.007
  })
  return (
    <mesh
      ref={ref}
      position={position}
      onClick={() => onSelect(data)}
      onPointerOver={() => document.body.style.cursor = 'pointer'}
      onPointerOut={() => document.body.style.cursor = 'default'}
    >
      <dodecahedronGeometry args={[0.15, 0]} />
      <meshStandardMaterial
        color={data.is_potentially_hazardous ? '#ff3300' : '#888888'}
        roughness={0.9}
        metalness={0.2}
      />
    </mesh>
  )
}

function CameraController({ target, onArrived }) {
  const { camera } = useThree()
  const keys = useRef({})
  const mouse = useRef({ active: false, x: 0, y: 0 })
  const yaw = useRef(Math.PI)
  const pitch = useRef(-0.3)
  const travel = useRef(null)
  const arrived = useRef(false)

  useEffect(() => {
    if (target) {
      travel.current = {
        from: camera.position.clone(),
        to: new THREE.Vector3(target.x + target.size * 4, target.size * 2, target.z + target.size * 4),
        t: 0
      }
      arrived.current = false
    }
  }, [target])

  useEffect(() => {
    const kd = e => { keys.current[e.code] = true }
    const ku = e => { keys.current[e.code] = false }
    const md = e => { mouse.current = { active: true, x: e.clientX, y: e.clientY } }
    const mu = () => { mouse.current.active = false }
    const mm = e => {
      if (!mouse.current.active) return
      yaw.current -= (e.clientX - mouse.current.x) * 0.003
      pitch.current = Math.max(-1.4, Math.min(1.4, pitch.current - (e.clientY - mouse.current.y) * 0.003))
      mouse.current.x = e.clientX
      mouse.current.y = e.clientY
    }
    window.addEventListener('keydown', kd)
    window.addEventListener('keyup', ku)
    window.addEventListener('mousedown', md)
    window.addEventListener('mouseup', mu)
    window.addEventListener('mousemove', mm)
    return () => {
      window.removeEventListener('keydown', kd)
      window.removeEventListener('keyup', ku)
      window.removeEventListener('mousedown', md)
      window.removeEventListener('mouseup', mu)
      window.removeEventListener('mousemove', mm)
    }
  }, [])

  useFrame((_, delta) => {
    if (travel.current && !arrived.current) {
      travel.current.t += delta * 0.5
      const t = Math.min(travel.current.t, 1)
      const ease = t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t
      camera.position.lerpVectors(travel.current.from, travel.current.to, ease)
      if (target) camera.lookAt(target.x, 0, target.z)
      if (t >= 1) {
        arrived.current = true
        travel.current = null
        onArrived?.()
      }
      return
    }

    const speed = keys.current['ShiftLeft'] ? 0.8 : 0.25
    const dir = new THREE.Vector3(
      Math.cos(pitch.current) * Math.sin(yaw.current),
      Math.sin(pitch.current),
      Math.cos(pitch.current) * Math.cos(yaw.current)
    ).normalize()
    const right = new THREE.Vector3().crossVectors(dir, new THREE.Vector3(0, 1, 0)).normalize()

    if (keys.current['KeyW'] || keys.current['ArrowUp'])    camera.position.addScaledVector(dir, speed)
    if (keys.current['KeyS'] || keys.current['ArrowDown'])  camera.position.addScaledVector(dir, -speed)
    if (keys.current['KeyA'] || keys.current['ArrowLeft'])  camera.position.addScaledVector(right, -speed)
    if (keys.current['KeyD'] || keys.current['ArrowRight']) camera.position.addScaledVector(right, speed)
    if (keys.current['Space'])  camera.position.y += speed
    if (keys.current['KeyQ'])   camera.position.y -= speed

    camera.lookAt(camera.position.clone().add(dir))
  })

  return null
}

function Toggle({ on, onChange, color }) {
  return (
    <div
      onClick={() => onChange(v => !v)}
      style={{
        width: 36, height: 20, borderRadius: 10, cursor: 'pointer',
        background: on ? color : 'rgba(255,255,255,0.2)',
        position: 'relative', transition: 'background 0.2s', flexShrink: 0
      }}
    >
      <div style={{
        position: 'absolute', top: 2, left: on ? 18 : 2,
        width: 16, height: 16, borderRadius: '50%', background: 'white',
        transition: 'left 0.2s'
      }} />
    </div>
  )
}

export default function App() {
  const [selected, setSelected]           = useState(null)
  const [asteroids, setAsteroids]         = useState([])
  const [showAsteroids, setShowAsteroids] = useState(true)
  const [showISS, setShowISS]             = useState(true)
  const [showML, setShowML]               = useState(false)
  const [simDate, setSimDate]             = useState(new Date().toISOString().split('T')[0])
  const [travelTarget, setTravelTarget]   = useState(null)
  const [arrivedMsg, setArrivedMsg]       = useState('')
  const [asteroidCount, setAsteroidCount] = useState(0)

  const jd = dateToJD(simDate)

  const positions = useMemo(() => {
    const p = {}
    PLANET_DATA.forEach(d => { p[d.name] = keplerPos(d.name, jd) })
    return p
  }, [jd])

  const alignments = useMemo(() => detectAlignments(positions), [positions])
  const historyEvents = useMemo(() => getHistoryForDate(simDate), [simDate])
  console.log('history check:', simDate, historyEvents)


  useEffect(() => {
    const today = new Date().toISOString().split('T')[0]
    fetch(`https://api.nasa.gov/neo/rest/v1/feed?start_date=${today}&end_date=${today}&api_key=${NASA_KEY}`)
      .then(r => r.json())
      .then(data => {
        const neos = Object.values(data.near_earth_objects).flat().slice(0, 12)
        setAsteroidCount(neos.length)
        setAsteroids(neos.map((neo, i) => ({
          id: neo.id,
          name: neo.name,
          is_potentially_hazardous: neo.is_potentially_hazardous_asteroid,
          diameter: neo.estimated_diameter.kilometers,
          velocity: neo.close_approach_data[0]?.relative_velocity.kilometers_per_hour,
          miss_distance: neo.close_approach_data[0]?.miss_distance.kilometers,
          position: [
            10 + Math.cos((i / 12) * Math.PI * 2) * 6,
            (Math.random() - 0.5) * 3,
            Math.sin((i / 12) * Math.PI * 2) * 6,
          ]
        })))
      }).catch(() => {})
  }, [])

  const travelTo = name => {
    if (name === 'Sun') {
      setTravelTarget({ x: 0, y: 0, z: 0, size: 2.5, name: 'Sun' })
      setSelected({ name: 'Sun', info: 'Our star. 1.4 million km wide. Surface 5,500°C, core 15 million°C.' })
    } else {
      const data = PLANET_DATA.find(p => p.name === name)
      const pos = positions[name]
      setTravelTarget({ x: pos.x, y: 0, z: pos.z, size: data.size, name })
      setSelected({ name, info: data.info })
    }
    setArrivedMsg('')
  }

  const handleAsteroidSelect = data => {
    const size = parseFloat(data.diameter?.estimated_diameter_max ?? 0.1)
    const vel  = parseFloat(data.velocity ?? 30000)
    const dist = parseFloat(data.miss_distance ?? 300000)
    setSelected({ ...data, knnThreat: knnClassify(size, vel, dist) })
  }

  return (
    <div style={{ width: '100vw', height: '100vh', background: 'black', position: 'relative', fontFamily: 'sans-serif' }}>
      <Canvas
        camera={{ position: [0, 15, 25], fov: 60 }}
        gl={{ antialias: true, powerPreference: 'high-performance' }}
      >
        <ambientLight intensity={0.35} />
        <pointLight position={[0, 0, 0]} intensity={6} distance={400} decay={0.5} />
        <pointLight position={[0, 20, 0]} intensity={0.8} distance={400} decay={0.5} />
        <Stars radius={300} depth={80} count={10000} factor={5} />
        <Sun onClick={() => travelTo('Sun')} />
        {PLANET_DATA.map(p => (
          <OrbitRing key={p.name + '-ring'} position={positions[p.name]} />
        ))}
        {PLANET_DATA.map(p => (
          <Planet
            key={p.name}
            data={p}
            position={positions[p.name]}
            onSelect={d => travelTo(d.name)}
            isTarget={travelTarget?.name === p.name}
          />
        ))}
        {showAsteroids && asteroids.map(a => (
          <Asteroid key={a.id} position={a.position} data={a} onSelect={handleAsteroidSelect} />
        ))}
        {showISS && <ISS onSelect={setSelected} />}
        <CameraController
          target={travelTarget}
          onArrived={() => setArrivedMsg(travelTarget?.name ?? '')}
        />
      </Canvas>

      {/* title */}
      <div style={{ position: 'absolute', top: 24, left: 32, pointerEvents: 'none' }}>
        <div style={{ fontSize: 28, fontWeight: 700, letterSpacing: 4, color: 'white' }}>COSMORA</div>
        <div style={{ fontSize: 11, opacity: 0.4, letterSpacing: 3, color: 'white' }}>SOLAR SYSTEM EXPLORER</div>
        <div style={{ fontSize: 10, opacity: 0.3, color: 'white', marginTop: 4 }}>
          live NASA data · real orbital mechanics · KNN threat classifier
        </div>
      </div>

      {/* planetary alignment alert */}
      {alignments.length > 0 && (
        <div style={{
          position: 'absolute', top: 24, left: '50%', transform: 'translateX(-50%)',
          background: 'rgba(255,200,0,0.12)', border: '1px solid rgba(255,200,0,0.4)',
          borderRadius: 8, padding: '8px 18px', color: '#ffcc00', fontSize: 12,
          backdropFilter: 'blur(8px)', textAlign: 'center', maxWidth: 400, zIndex: 100
        }}>
          ✦ rare alignment detected — {alignments[0].join(', ')} ✦
          {alignments.length > 1 && (
            <div style={{ fontSize: 10, opacity: 0.7, marginTop: 2 }}>
              +{alignments.length - 1} more on this date
            </div>
          )}
        </div>
      )}

      {/* space history banner */}
      {historyEvents.length > 0 && (
        <div style={{
          position: 'absolute',
          top: alignments.length > 0 ? 90 : 24,
          left: '50%', transform: 'translateX(-50%)',
          background: 'rgba(100,180,255,0.08)', border: '1px solid rgba(100,180,255,0.25)',
          borderRadius: 8, padding: '8px 18px', color: '#88ccff', fontSize: 11,
          backdropFilter: 'blur(8px)', textAlign: 'center', maxWidth: 440, zIndex: 100
        }}>
          {historyEvents.map((e, i) => <div key={i}>🚀 {e.text}</div>)}
        </div>
      )}

      {/* controls */}
      <div style={{
        position: 'absolute', top: 24, right: 32, color: 'white',
        fontSize: 12, display: 'flex', flexDirection: 'column', gap: 10, alignItems: 'flex-end'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ opacity: 0.5 }}>time machine</span>
          <input
            type="date"
            value={simDate}
            onChange={e => setSimDate(e.target.value)}
            style={{
              background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.15)',
              color: 'white', borderRadius: 6, padding: '3px 8px', fontSize: 11, colorScheme: 'dark'
            }}
          />
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ opacity: 0.5 }}>
            near-earth asteroids
            {asteroidCount > 0 && <span style={{ marginLeft: 6, color: '#ff6644' }}>({asteroidCount} today)</span>}
          </span>
          <Toggle on={showAsteroids} onChange={setShowAsteroids} color="#ff4400" />
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ opacity: 0.5 }}>ISS tracker</span>
          <Toggle on={showISS} onChange={setShowISS} color="#4488ff" />
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ opacity: 0.5 }}>KNN classifier</span>
          <Toggle on={showML} onChange={setShowML} color="#aa44ff" />
        </div>
        <div style={{ opacity: 0.25, fontSize: 10, textAlign: 'right', lineHeight: 1.9, marginTop: 2 }}>
          <span style={{ color: '#ff3300' }}>■</span> hazardous &nbsp;
          <span style={{ color: '#888' }}>■</span> safe asteroid<br />
          click planet to travel · WASD to fly<br />
          drag to look · shift = fast · Q = down
        </div>
      </div>

      {/* KNN info panel */}
      {showML && (
        <div style={{
          position: 'absolute', top: 220, right: 32,
          background: 'rgba(80,0,180,0.15)', border: '1px solid rgba(150,80,255,0.3)',
          borderRadius: 10, padding: '14px 18px', color: 'white',
          fontSize: 11, maxWidth: 220, backdropFilter: 'blur(12px)'
        }}>
          <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8, color: '#cc88ff' }}>KNN Threat Classifier</div>
          <div style={{ opacity: 0.6, lineHeight: 1.7, marginBottom: 10 }}>
            K-nearest-neighbours (k=3) trained on 14 labeled asteroids. Features: size, velocity, miss distance.
          </div>
          <div style={{ opacity: 0.5, fontSize: 10, marginBottom: 6 }}>threat levels</div>
          {['low', 'medium', 'high', 'extreme'].map(t => (
            <div key={t} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
              <div style={{ width: 8, height: 8, borderRadius: '50%', background: THREAT_COLORS[t], flexShrink: 0 }} />
              <span style={{ opacity: 0.8 }}>{t}</span>
            </div>
          ))}
          <div style={{ opacity: 0.4, fontSize: 10, marginTop: 8 }}>click an asteroid to classify</div>
        </div>
      )}

      {/* planet nav */}
      <div style={{ position: 'absolute', bottom: 40, right: 32, display: 'flex', flexDirection: 'column', gap: 4 }}>
        <div style={{ fontSize: 10, opacity: 0.3, color: 'white', marginBottom: 4, textAlign: 'right', letterSpacing: 2 }}>
          NAVIGATE
        </div>
        {['Sun', ...PLANET_DATA.map(p => p.name)].map(name => {
          const pdata = PLANET_DATA.find(p => p.name === name)
          const isActive = travelTarget?.name === name
          return (
            <button key={name} onClick={() => travelTo(name)} style={{
              background: isActive ? 'rgba(255,255,255,0.12)' : 'rgba(0,0,0,0.4)',
              border: `1px solid ${isActive ? 'rgba(255,255,255,0.4)' : 'rgba(255,255,255,0.1)'}`,
              color: 'white', borderRadius: 6, padding: '5px 12px', cursor: 'pointer', fontSize: 11,
              display: 'flex', alignItems: 'center', gap: 8, justifyContent: 'flex-end',
              transition: 'all 0.2s', backdropFilter: 'blur(4px)'
            }}>
              <span style={{ opacity: 0.4, fontSize: 10 }}>{pdata?.distAU ?? '—'}</span>
              <span style={{ fontWeight: isActive ? 500 : 400 }}>{name}</span>
              <div style={{
                width: 8, height: 8, borderRadius: '50%', flexShrink: 0,
                background: name === 'Sun' ? '#ff8800' : (pdata?.color ?? '#fff')
              }} />
            </button>
          )
        })}
      </div>

      {/* arrived message */}
      {arrivedMsg && (
        <div style={{
          position: 'absolute', top: '42%', left: '50%', transform: 'translate(-50%, -50%)',
          color: 'white', fontSize: 13, textAlign: 'center', pointerEvents: 'none', opacity: 0.7
        }}>
          arrived at {arrivedMsg}<br />
          <span style={{ fontSize: 10, opacity: 0.5 }}>WASD + drag to explore · click nav to travel</span>
        </div>
      )}

      {/* info panel */}
      {selected && (
        <div style={{
          position: 'absolute', bottom: 40, left: 32,
          background: 'rgba(0,0,0,0.88)', border: '1px solid rgba(255,255,255,0.1)',
          borderRadius: 12, padding: '18px 24px', color: 'white', maxWidth: 340,
          backdropFilter: 'blur(16px)'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 10 }}>
            <div>
              <div style={{ fontSize: 20, fontWeight: 600 }}>{selected.name}</div>
              {selected.is_potentially_hazardous && (
                <span style={{ fontSize: 10, color: '#ff3300', border: '1px solid #ff3300', borderRadius: 4, padding: '1px 6px', marginTop: 4, display: 'inline-block' }}>
                  POTENTIALLY HAZARDOUS
                </span>
              )}
              {selected.knnThreat && (
                <div style={{ marginTop: 6, display: 'flex', alignItems: 'center', gap: 6 }}>
                  <div style={{ width: 8, height: 8, borderRadius: '50%', background: THREAT_COLORS[selected.knnThreat] }} />
                  <span style={{ fontSize: 11, color: THREAT_COLORS[selected.knnThreat] }}>
                    KNN threat: {selected.knnThreat}
                  </span>
                </div>
              )}
            </div>
            <button onClick={() => setSelected(null)} style={{
              background: 'transparent', border: 'none', color: 'rgba(255,255,255,0.4)',
              cursor: 'pointer', fontSize: 18, lineHeight: 1, padding: 0, marginLeft: 12
            }}>×</button>
          </div>

          <div style={{ fontSize: 13, opacity: 0.8, lineHeight: 1.7, marginBottom: 10 }}>{selected.info}</div>

          {PLANET_DATA.find(p => p.name === selected.name) && (() => {
            const pd = PLANET_DATA.find(p => p.name === selected.name)
            return (
              <div style={{ display: 'flex', gap: 16, fontSize: 11, opacity: 0.5, borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: 10 }}>
                <div><div style={{ opacity: 0.6 }}>distance</div><div>{pd.distAU}</div></div>
                <div><div style={{ opacity: 0.6 }}>moons</div><div>{pd.moons}</div></div>
              </div>
            )
          })()}

          {selected.diameter && (
            <div style={{ fontSize: 12, opacity: 0.7, lineHeight: 2, borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: 10 }}>
              <div>diameter: {parseFloat(selected.diameter.estimated_diameter_min).toFixed(3)} – {parseFloat(selected.diameter.estimated_diameter_max).toFixed(3)} km</div>
              <div>velocity: {parseFloat(selected.velocity).toLocaleString()} km/h</div>
              <div>miss distance: {parseFloat(selected.miss_distance).toLocaleString()} km</div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
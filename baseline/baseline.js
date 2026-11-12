// Guided test that measures how the 8th Wall engine tracks on this phone: jitter while
// still, drift over a walked loop, recovery after the lens is covered, frame cost and memory.
/* global XR8 */
import * as THREE from 'three'
import {
  summarizeFrames, summarizeHoldStill, summarizeLensCover, summarizeLoop,
} from './measurements.js'

window.THREE = THREE  // XR8.Threejs reads three.js from the global.

const TESTS = [
  {
    key: 'holdStill',
    title: 'Hold still',
    instruction: 'Prop the phone on your start mark so it cannot move, camera facing the room. ' +
      'Tap Begin and leave it alone for 10 seconds.',
    seconds: 10,
    summarize: summarizeHoldStill,
  },
  {
    key: 'loop',
    title: 'Walk a loop',
    instruction: 'With the phone on the start mark, tap Begin. Pick it up, walk your loop, put ' +
      'it back on the mark exactly as it was, then tap End.',
    summarize: summarizeLoop,
  },
  {
    key: 'lensCover',
    title: 'Cover the lens',
    instruction: 'Leave the phone on the mark. Tap Begin, cover the camera for 3 seconds, ' +
      'uncover it, wait for the picture to settle, then tap End.',
    summarize: summarizeLensCover,
  },
]

const card = document.getElementById('card')
const canvas = document.getElementById('camera')

const samples = []
const memorySamples = []
const session = {
  page: 'slam-baseline',
  engine: '8th Wall XR Engine binary',
  startedAt: new Date().toISOString(),
  userAgent: navigator.userAgent,
  devicePixelRatio: window.devicePixelRatio,
  webgpuAvailable: 'gpu' in navigator,
  tests: {},
}

let runStartedAt = 0
let trackerGpuStepMs = null
let trackerCpuStepMs = null
let marker = null
let camera = null

// ---------------------------------------------------------------------------------------
// Recording

const megabytes = bytes => bytes / (1024 * 1024)

const wasmHeapBytes = () =>
  window.capturedWasmMemories.reduce((sum, memory) => sum + memory.buffer.byteLength, 0)

function recordMemory() {
  memorySamples.push({
    seconds: (performance.now() - runStartedAt) / 1000,
    wasmHeapBytes: wasmHeapBytes(),
    // Chrome only.
    jsHeapBytes: performance.memory ? performance.memory.usedJSHeapSize : null,
  })
}

// The tracker does its work in two callbacks per camera frame: onProcessGpu (feature
// extraction on the GPU and reading the result back) and onProcessCpu (solving the pose).
// Timing them separately shows which half a slow frame comes from.
function timedTracker() {
  const tracker = XR8.XrController.pipelineModule()
  const {onProcessGpu, onProcessCpu} = tracker
  tracker.onProcessGpu = (input) => {
    const startedAt = performance.now()
    const result = onProcessGpu(input)
    trackerGpuStepMs = performance.now() - startedAt
    return result
  }
  tracker.onProcessCpu = (input) => {
    const startedAt = performance.now()
    const result = onProcessCpu(input)
    trackerCpuStepMs = performance.now() - startedAt
    return result
  }
  return tracker
}

const recorderModule = {
  name: 'baseline-recorder',
  onStart: ({GLctx, videoWidth, videoHeight, canvasWidth, canvasHeight}) => {
    const rendererInfo = GLctx.getExtension('WEBGL_debug_renderer_info')
    Object.assign(session, {
      engineVersion: XR8.version(),
      threeRevision: THREE.REVISION,
      webglVersion: GLctx.getParameter(GLctx.VERSION),
      gpu: rendererInfo ? GLctx.getParameter(rendererInfo.UNMASKED_RENDERER_WEBGL) : null,
      videoWidth,
      videoHeight,
      canvasWidth,
      canvasHeight,
    })
  },
  onUpdate: ({processCpuResult}) => {
    const {reality} = processCpuResult
    // The engine reports no pose until its first camera frame has been processed.
    if (!reality || !reality.position) {
      return
    }
    const {position, rotation} = reality
    samples.push({
      seconds: (performance.now() - runStartedAt) / 1000,
      position: {x: position.x, y: position.y, z: position.z},
      rotation: {x: rotation.x, y: rotation.y, z: rotation.z, w: rotation.w},
      tracking: reality.trackingStatus,
      reason: reality.trackingReason,
      worldPoints: reality.worldPoints ? reality.worldPoints.length : 0,
      trackerGpuStepMs,
      trackerCpuStepMs,
    })
    // Stay null until the tracker next processes a camera frame.
    trackerGpuStepMs = null
    trackerCpuStepMs = null
    if (session.secondsToFirstTracking === undefined && reality.trackingStatus === 'NORMAL') {
      session.secondsToFirstTracking = samples.at(-1).seconds
      showTest(0)
    }
  },
  onCameraStatusChange: ({status}) => {
    if (status === 'failed') {
      showStartFailure('the camera did not start')
    }
  },
  onException: (error) => {
    showStartFailure(error.permission ? `${error.permission} access was ${error.status}` : String(error.message || error))
  },
}

const sceneModule = {
  name: 'baseline-scene',
  onStart: () => {
    const xrScene = XR8.Threejs.xrScene()
    camera = xrScene.camera
    // In responsive scale the engine treats this height as the camera's real height above
    // the floor, so one scene unit is the phone's starting height.
    camera.position.set(0, 1, 0)
    marker = new THREE.AxesHelper(0.1)
    marker.visible = false
    xrScene.scene.add(marker)
    XR8.XrController.updateCameraProjectionMatrix({
      origin: camera.position,
      facing: camera.quaternion,
    })
  },
}

// Leaves a fixed object in view so drift can also be seen, not only measured.
function dropMarker() {
  marker.position.copy(camera.position).addScaledVector(camera.getWorldDirection(new THREE.Vector3()), 0.5)
  marker.visible = true
}

// ---------------------------------------------------------------------------------------
// Screens

function showCard({step, title, text, html = '', buttons = []}) {
  card.innerHTML = `
    ${step ? `<p class="step">${step}</p>` : ''}
    <h1>${title}</h1>
    ${text ? `<p>${text}</p>` : ''}
    ${html}`
  for (const {label, onTap, secondary} of buttons) {
    const button = document.createElement('button')
    button.textContent = label
    button.className = secondary ? 'secondary' : ''
    button.onclick = () => onTap(button)
    card.append(button)
  }
}

function showIntro() {
  showCard({
    title: 'SLAM baseline',
    text: 'Mark a spot on a table or shelf where the phone can be propped upright. Every ' +
      'test starts and ends with the phone on that mark.',
    html: `
      <label><input type="checkbox" id="absolute-scale">Measure in metres (slower to start)</label>
      <p class="licence">World tracking by the 8th Wall XR Engine, © Niantic Spatial, Inc.
        <a href="xr/LICENSE">Licence</a></p>`,
    buttons: [{label: 'Start camera', onTap: startCamera}],
  })
}

function showStartFailure(reason) {
  showCard({
    title: 'Could not start',
    text: `The engine reported: ${reason}. This test needs a phone with camera and motion ` +
      'access allowed for this site. Change that in the browser settings, then reload.',
  })
}

// iOS only shows its motion-access prompt when asked directly from a tap, so this runs
// before the engine starts. Other browsers have no such prompt.
async function requestMotionAccess() {
  for (const sensorEvent of [window.DeviceMotionEvent, window.DeviceOrientationEvent]) {
    if (sensorEvent && sensorEvent.requestPermission) {
      // A refusal is not handled here: the engine reports it when it starts.
      await sensorEvent.requestPermission().catch(() => {})
    }
  }
}

async function startCamera() {
  await requestMotionAccess()
  session.scale = document.getElementById('absolute-scale').checked ? 'absolute' : 'responsive'
  showCard({
    title: 'Finding the room',
    text: 'Point the camera at the room and move the phone slowly from side to side.',
  })
  XR8.XrController.configure({enableWorldPoints: true, scale: session.scale})
  XR8.addCameraPipelineModules([
    XR8.GlTextureRenderer.pipelineModule(),
    XR8.Threejs.pipelineModule(),
    timedTracker(),
    sceneModule,
    recorderModule,
  ])
  runStartedAt = performance.now()
  XR8.run({canvas})
  document.getElementById('readout').hidden = false
  setInterval(recordMemory, 1000)
  setInterval(updateReadout, 250)
}

function showTest(index) {
  const test = TESTS[index]
  const step = `Test ${index + 1} of ${TESTS.length}`
  const finish = (firstSample) => {
    session.tests[test.key] = test.summarize(samples.slice(firstSample))
    if (index + 1 < TESTS.length) {
      showTest(index + 1)
    } else {
      showResults()
    }
  }
  const begin = () => {
    const firstSample = samples.length
    dropMarker()
    if (test.seconds) {
      showCard({step, title: test.title, text: `Recording for ${test.seconds} seconds. Do not touch the phone.`})
      setTimeout(() => finish(firstSample), test.seconds * 1000)
    } else {
      showCard({
        step,
        title: test.title,
        text: test.instruction,
        buttons: [{label: 'End', onTap: () => finish(firstSample)}],
      })
    }
  }
  showCard({step, title: test.title, text: test.instruction, buttons: [{label: 'Begin', onTap: begin}]})
}

// ---------------------------------------------------------------------------------------
// Results

function resultSections() {
  const length = session.scale === 'absolute' ? 'm' : 'units'
  const {holdStill, loop, lensCover} = session.tests
  const {frames, memory} = session
  const number = (value, digits, unit) => `${value.toFixed(digits)} ${unit}`
  return [
    ['Walk a loop', [
      ['Drift, share of path', number(loop.closureErrorPercentOfPath, 2, '%')],
      ['Drift, distance', number(loop.closureError, 3, length)],
      ['Drift, rotation', number(loop.closureRotationDegrees, 2, '°')],
      ['Path walked', number(loop.pathLength, 2, length)],
      ['Time not tracking', number(loop.secondsNotTracking, 1, 's')],
    ]],
    ['Hold still', [
      ['Position jitter (RMS)', number(holdStill.positionJitterRms, 4, length)],
      ['Position jitter (max)', number(holdStill.positionJitterMax, 4, length)],
      ['Rotation jitter (RMS)', number(holdStill.rotationJitterRmsDegrees, 3, '°')],
      ['Creep over 10 s', number(holdStill.positionCreep, 4, length)],
    ]],
    ['Cover the lens', [
      ['Time not tracking', number(lensCover.secondsNotTracking, 1, 's')],
      ['Position jump', number(lensCover.positionJump, 3, length)],
      ['Rotation jump', number(lensCover.rotationJumpDegrees, 2, '°')],
    ]],
    ['Speed', [
      ['Time to first tracking', number(session.secondsToFirstTracking, 1, 's')],
      ['Display frames', number(frames.displayFramesPerSecond, 1, 'fps')],
      ['Tracker frames', number(frames.trackerFramesPerSecond, 1, 'fps')],
      ['Tracker time (mean)', number(frames.trackerMeanMs, 1, 'ms')],
      ['Tracker time (95th)', number(frames.trackerP95Ms, 1, 'ms')],
      ['Tracker time (max)', number(frames.trackerMaxMs, 1, 'ms')],
      ['GPU step (mean)', number(frames.trackerGpuStepMeanMs, 1, 'ms')],
      ['CPU step (mean)', number(frames.trackerCpuStepMeanMs, 1, 'ms')],
      ['Frame interval (95th)', number(frames.frameIntervalP95Ms, 1, 'ms')],
      ['Stalls over 50 ms', `${frames.framesOver50Ms}`],
    ]],
    ['Memory', [
      ['Wasm heap at start', number(memory.wasmHeapStartMegabytes, 0, 'MB')],
      ['Wasm heap at end', number(memory.wasmHeapEndMegabytes, 0, 'MB')],
      ['Map points (mean)', number(frames.worldPointsMean, 0, '')],
    ]],
    ['Device', [
      ['WebGPU available', session.webgpuAvailable ? 'yes' : 'no'],
    ]],
  ]
}

function showResults() {
  session.frames = summarizeFrames(samples)
  session.memory = {
    wasmHeapStartMegabytes: megabytes(memorySamples[0].wasmHeapBytes),
    wasmHeapEndMegabytes: megabytes(memorySamples.at(-1).wasmHeapBytes),
  }
  const sections = resultSections()
  const rows = sections.map(([heading, lines]) =>
    `<tr><th colspan="2">${heading}</th></tr>` +
    lines.map(([label, value]) => `<tr><td>${label}</td><td>${value}</td></tr>`).join('')
  ).join('')
  const summaryText = [
    `SLAM baseline, ${session.engine} ${session.engineVersion}, scale ${session.scale}`,
    session.userAgent,
    `${session.gpu}, video ${session.videoWidth}x${session.videoHeight}`,
    ...sections.flatMap(([heading, lines]) =>
      [`[${heading}]`, ...lines.map(([label, value]) => `${label}: ${value}`)]),
  ].join('\n')
  showCard({
    title: 'Results',
    html: `<table>${rows}</table>`,
    buttons: [
      {label: 'Download full recording', onTap: downloadRecording},
      {
        label: 'Copy summary',
        secondary: true,
        onTap: async (button) => {
          await navigator.clipboard.writeText(summaryText)
          button.textContent = 'Copied'
        },
      },
    ],
  })
}

function downloadRecording() {
  const recording = JSON.stringify({...session, samples, memorySamples})
  const link = document.createElement('a')
  link.href = URL.createObjectURL(new Blob([recording], {type: 'application/json'}))
  link.download = `slam-baseline-${session.startedAt.replace(/[:.]/g, '-')}.json`
  link.click()
}

function updateReadout() {
  const recent = samples.slice(-60)
  if (recent.length < 2) {
    return
  }
  const latest = recent.at(-1)
  const tracking = document.getElementById('readout-tracking')
  tracking.textContent = latest.tracking === 'NORMAL' ? 'Normal' : (latest.reason || latest.tracking).toLowerCase().replaceAll('_', ' ')
  tracking.className = latest.tracking === 'NORMAL' ? 'good' : 'warning'
  const fps = (recent.length - 1) / (latest.seconds - recent[0].seconds)
  document.getElementById('readout-fps').textContent = fps.toFixed(0)
  const trackerFrames = recent.filter(sample => sample.trackerCpuStepMs !== null)
  if (trackerFrames.length) {
    const total = trackerFrames.reduce(
      (sum, sample) => sum + sample.trackerGpuStepMs + sample.trackerCpuStepMs, 0
    )
    document.getElementById('readout-tracker').textContent = (total / trackerFrames.length).toFixed(1)
  }
  document.getElementById('readout-points').textContent = latest.worldPoints
  document.getElementById('readout-memory').textContent = megabytes(wasmHeapBytes()).toFixed(0)
}

showCard({title: 'Loading the engine'})
if (window.XR8) {
  showIntro()
} else {
  window.addEventListener('xrloaded', showIntro)
}

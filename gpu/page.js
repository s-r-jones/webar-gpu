// Page around the GPU feature test: camera start, live readout, copyable numbers.
import {startFeatureTest} from './gpu.js'

const card = document.getElementById('card')
const video = document.getElementById('video')
const recentFrames = []
let testInfo = null

function showCard({title, text, button}) {
  card.innerHTML = `<h1>${title}</h1><p>${text}</p>`
  if (button) {
    const element = document.createElement('button')
    element.textContent = button.label
    element.onclick = () => button.onTap(element)
    card.append(element)
  }
}

const mean = values => values.reduce((sum, value) => sum + value, 0) / values.length

function averages() {
  const seconds = (recentFrames.at(-1).at - recentFrames[0].at) / 1000
  return {
    keypoints: mean(recentFrames.map(frame => frame.keypointCount)),
    gpuMs: testInfo.canTimeGpu ? mean(recentFrames.map(frame => frame.gpuMs)) : null,
    mainThreadMs: mean(recentFrames.map(frame => frame.mainThreadMs)),
    resultsAfterMs: mean(recentFrames.map(frame => frame.resultsAfterMs)),
    framesPerSecond: (recentFrames.length - 1) / seconds,
    readbackKilobytes: recentFrames[0].readbackKilobytes,
  }
}

function updateReadout() {
  if (recentFrames.length < 2) {
    return
  }
  const now = averages()
  document.getElementById('readout-keypoints').textContent = now.keypoints.toFixed(0)
  document.getElementById('readout-gpu').textContent = now.gpuMs === null ? 'n/a' : now.gpuMs.toFixed(1)
  document.getElementById('readout-main').textContent = now.mainThreadMs.toFixed(1)
  document.getElementById('readout-results').textContent = now.resultsAfterMs.toFixed(0)
  document.getElementById('readout-fps').textContent = now.framesPerSecond.toFixed(0)
}

async function copyNumbers(button) {
  const now = averages()
  await navigator.clipboard.writeText([
    'GPU feature test',
    navigator.userAgent,
    `Camera ${video.videoWidth}x${video.videoHeight}, pyramid ${testInfo.levels.map(level => `${level.width}x${level.height}`).join(' ')}`,
    `Keypoints per frame: ${now.keypoints.toFixed(0)} of ${testInfo.slotCount} cells`,
    `GPU time per frame: ${now.gpuMs === null ? 'not measurable on this browser' : `${now.gpuMs.toFixed(2)} ms`}`,
    `Main-thread time per frame: ${now.mainThreadMs.toFixed(2)} ms`,
    `Results back on the CPU after: ${now.resultsAfterMs.toFixed(1)} ms`,
    `Readback per frame: ${now.readbackKilobytes.toFixed(0)} KB`,
    `Frames per second: ${now.framesPerSecond.toFixed(1)}`,
  ].join('\n'))
  button.textContent = 'Copied'
}

async function start() {
  showCard({title: 'Starting the camera', text: ''})
  video.srcObject = await navigator.mediaDevices.getUserMedia({
    video: {facingMode: 'environment', width: {ideal: 960}, height: {ideal: 720}},
  })
  await video.play()
  testInfo = await startFeatureTest({
    canvas: document.getElementById('camera'),
    video,
    onStats: (stats) => {
      recentFrames.push({...stats, at: performance.now()})
      if (recentFrames.length > 120) {
        recentFrames.shift()
      }
    },
    onError: message => showCard({title: 'GPU error', text: message}),
  })
  document.getElementById('readout').hidden = false
  setInterval(updateReadout, 250)
  showCard({
    title: 'GPU feature test',
    text: `The dots are corners found and described on the GPU for every camera frame, up to ${testInfo.slotCount} a frame. ` +
      'For comparison, 8th Wall\'s feature step on this iPhone took about 4.1 ms on the main thread ' +
      'and reads 4,096 KB back from the GPU every frame.',
    button: {label: 'Copy numbers', onTap: copyNumbers},
  })
}

if (navigator.gpu) {
  showCard({
    title: 'GPU feature test',
    text: 'Finds and describes image corners on the GPU using the live camera, and shows what it costs per frame.',
    button: {label: 'Start camera', onTap: () => start().catch(error => showCard({title: 'Could not start', text: error.message}))},
  })
} else {
  showCard({title: 'WebGPU is not available', text: 'This browser does not offer WebGPU, so the test cannot run here.'})
}

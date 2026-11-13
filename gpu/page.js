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
    matched: mean(recentFrames.map(frame => frame.matchedCount)),
    longTracks: mean(recentFrames.map(frame => frame.longTrackCount)),
    bitsDifferentPerMatch: mean(recentFrames.map(frame => frame.bitsDifferentPerMatch)),
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
  document.getElementById('readout-matched').textContent = now.matched.toFixed(0)
  document.getElementById('readout-long').textContent = now.longTracks.toFixed(0)
  document.getElementById('readout-gpu').textContent = now.gpuMs === null ? 'n/a' : now.gpuMs.toFixed(1)
  document.getElementById('readout-main').textContent = now.mainThreadMs.toFixed(1)
}

async function copyNumbers(button) {
  const now = averages()
  await navigator.clipboard.writeText([
    'GPU feature test',
    navigator.userAgent,
    `Camera ${video.videoWidth}x${video.videoHeight}, pyramid ${testInfo.levels.map(level => `${level.width}x${level.height}`).join(' ')}`,
    `Keypoints per frame: ${now.keypoints.toFixed(0)} of ${testInfo.slotCount} cells`,
    `Matched to the previous frame: ${now.matched.toFixed(0)} (${(100 * now.matched / now.keypoints).toFixed(0)}%)`,
    `Followed for half a second or more: ${now.longTracks.toFixed(0)} (${(100 * now.longTracks / now.keypoints).toFixed(0)}%)`,
    `Descriptor bits different per match: ${now.bitsDifferentPerMatch.toFixed(1)} of 256`,
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
    text: 'Corners are found, described and matched to the previous frame on the GPU. ' +
      'Green dots have been followed for half a second or more, yellow for less, red are new. ' +
      'Lines show how each followed corner moved since the last frame, drawn 8 times longer. ' +
      'Move the phone slowly, then tap Copy numbers.',
    button: {label: 'Copy numbers', onTap: copyNumbers},
  })
}

if (navigator.gpu) {
  showCard({
    title: 'GPU feature test',
    text: 'Finds image corners and follows them from frame to frame on the GPU using the live camera, and shows what it costs.',
    button: {label: 'Start camera', onTap: () => start().catch(error => showCard({title: 'Could not start', text: error.message}))},
  })
} else {
  showCard({title: 'WebGPU is not available', text: 'This browser does not offer WebGPU, so the test cannot run here.'})
}

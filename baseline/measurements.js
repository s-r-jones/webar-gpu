// Calculations over recorded pose samples. No browser APIs, so Node can test them.
//
// A sample is {seconds, position: {x, y, z}, rotation: {x, y, z, w}, tracking,
// worldPoints, trackerGpuStepMs, trackerCpuStepMs}. The two tracker times are null on
// display frames where the tracker had no new camera image to process.

export function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z)
}

export function angleBetweenRotationsDegrees(a, b) {
  const dot = Math.abs(a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w)
  return 2 * Math.acos(Math.min(1, dot)) * 180 / Math.PI
}

export function percentile(values, fraction) {
  const sorted = [...values].sort((a, b) => a - b)
  return sorted[Math.min(sorted.length - 1, Math.floor(fraction * sorted.length))]
}

const mean = values => values.reduce((sum, value) => sum + value, 0) / values.length

const rootMeanSquare = values => Math.sqrt(mean(values.map(value => value * value)))

const secondsSpanned = samples => samples.at(-1).seconds - samples[0].seconds

function meanPosition(samples) {
  return {
    x: mean(samples.map(sample => sample.position.x)),
    y: mean(samples.map(sample => sample.position.y)),
    z: mean(samples.map(sample => sample.position.z)),
  }
}

// Averaging quaternion components is only valid for rotations within a few degrees of
// each other, which is the case for a phone that is not being moved.
function meanRotation(samples) {
  const reference = samples[0].rotation
  const sum = {x: 0, y: 0, z: 0, w: 0}
  for (const {rotation} of samples) {
    const sameHemisphere = rotation.x * reference.x + rotation.y * reference.y +
      rotation.z * reference.z + rotation.w * reference.w >= 0
    const sign = sameHemisphere ? 1 : -1
    sum.x += sign * rotation.x
    sum.y += sign * rotation.y
    sum.z += sign * rotation.z
    sum.w += sign * rotation.w
  }
  const length = Math.hypot(sum.x, sum.y, sum.z, sum.w)
  return {x: sum.x / length, y: sum.y / length, z: sum.z / length, w: sum.w / length}
}

// Summing every frame-to-frame step would count jitter as travel, so steps are
// measured between poses at least a tenth of a second apart.
function pathLength(samples) {
  let length = 0
  let previous = samples[0]
  for (const sample of samples) {
    if (sample.seconds - previous.seconds >= 0.1) {
      length += distance(sample.position, previous.position)
      previous = sample
    }
  }
  return length
}

function secondsNotTracking(samples) {
  let seconds = 0
  for (let i = 0; i < samples.length - 1; i++) {
    if (samples[i].tracking !== 'NORMAL') {
      seconds += samples[i + 1].seconds - samples[i].seconds
    }
  }
  return seconds
}

// Ends the loop test without a tap: true once the phone has travelled at least 3 units
// and then moved less than 0.1 units over the last 2 seconds. One unit is roughly the
// phone's height above the floor, or a metre in absolute scale.
export function hasWalkedAndStopped(samples) {
  if (samples.length === 0 || pathLength(samples) < 3) {
    return false
  }
  const lastTwoSeconds = samples.filter(sample => sample.seconds > samples.at(-1).seconds - 2)
  return pathLength(lastTwoSeconds) < 0.1
}

// The phone is held still, so reported motion is tracker error plus hand tremor.
export function summarizeHoldStill(samples) {
  const centre = meanPosition(samples)
  const middleRotation = meanRotation(samples)
  const positionDeviations = samples.map(sample => distance(sample.position, centre))
  const rotationDeviations = samples.map(
    sample => angleBetweenRotationsDegrees(sample.rotation, middleRotation)
  )
  return {
    seconds: secondsSpanned(samples),
    positionJitterRms: rootMeanSquare(positionDeviations),
    positionJitterMax: Math.max(...positionDeviations),
    rotationJitterRmsDegrees: rootMeanSquare(rotationDeviations),
    positionCreep: distance(samples[0].position, samples.at(-1).position),
    rotationCreepDegrees: angleBetweenRotationsDegrees(samples[0].rotation, samples.at(-1).rotation),
    secondsNotTracking: secondsNotTracking(samples),
  }
}

// The phone starts and ends on the same physical mark, so the gap between the first
// and last pose is the drift accumulated over the walk.
export function summarizeLoop(samples) {
  const travelled = pathLength(samples)
  const closureError = distance(samples[0].position, samples.at(-1).position)
  return {
    seconds: secondsSpanned(samples),
    pathLength: travelled,
    closureError,
    closureErrorPercentOfPath: 100 * closureError / travelled,
    closureRotationDegrees: angleBetweenRotationsDegrees(samples[0].rotation, samples.at(-1).rotation),
    secondsNotTracking: secondsNotTracking(samples),
  }
}

// The phone stays on the mark while the lens is covered, so the pose should come back
// to where it was.
export function summarizeLensCover(samples) {
  return {
    seconds: secondsSpanned(samples),
    secondsNotTracking: secondsNotTracking(samples),
    positionJump: distance(samples[0].position, samples.at(-1).position),
    rotationJumpDegrees: angleBetweenRotationsDegrees(samples[0].rotation, samples.at(-1).rotation),
  }
}

export function summarizeFrames(samples) {
  const frameIntervals = samples.slice(1).map(
    (sample, i) => 1000 * (sample.seconds - samples[i].seconds)
  )
  const trackerFrames = samples.filter(sample => sample.trackerCpuStepMs !== null)
  const trackerTimes = trackerFrames.map(
    sample => sample.trackerGpuStepMs + sample.trackerCpuStepMs
  )
  return {
    seconds: secondsSpanned(samples),
    displayFramesPerSecond: frameIntervals.length / secondsSpanned(samples),
    trackerFramesPerSecond: trackerFrames.length / secondsSpanned(samples),
    frameIntervalMedianMs: percentile(frameIntervals, 0.5),
    frameIntervalP95Ms: percentile(frameIntervals, 0.95),
    frameIntervalMaxMs: percentile(frameIntervals, 1),
    // 50 ms is three missed refreshes on a 60 Hz screen: a stall the user can see.
    framesOver50Ms: frameIntervals.filter(interval => interval > 50).length,
    trackerMeanMs: mean(trackerTimes),
    trackerP95Ms: percentile(trackerTimes, 0.95),
    trackerMaxMs: percentile(trackerTimes, 1),
    trackerGpuStepMeanMs: mean(trackerFrames.map(sample => sample.trackerGpuStepMs)),
    trackerCpuStepMeanMs: mean(trackerFrames.map(sample => sample.trackerCpuStepMs)),
    worldPointsMean: mean(samples.map(sample => sample.worldPoints)),
    secondsNotTracking: secondsNotTracking(samples),
  }
}

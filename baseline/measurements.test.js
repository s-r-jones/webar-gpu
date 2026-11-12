// Run with: node --test
import assert from 'node:assert/strict'
import {test} from 'node:test'
import {
  angleBetweenRotationsDegrees, summarizeFrames, summarizeHoldStill, summarizeLensCover,
  summarizeLoop,
} from './measurements.js'

const identity = {x: 0, y: 0, z: 0, w: 1}

const rotationAboutY = (degrees) => {
  const half = degrees * Math.PI / 360
  return {x: 0, y: Math.sin(half), z: 0, w: Math.cos(half)}
}

const sample = (seconds, x, y, z, overrides = {}) => ({
  seconds,
  position: {x, y, z},
  rotation: identity,
  tracking: 'NORMAL',
  worldPoints: 100,
  trackerGpuStepMs: 2,
  trackerCpuStepMs: 3,
  ...overrides,
})

const close = (actual, expected, tolerance = 1e-9) =>
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} is not within ${tolerance} of ${expected}`)

test('angle between rotations is the rotation that separates them', () => {
  close(angleBetweenRotationsDegrees(identity, rotationAboutY(30)), 30, 1e-9)
  // q and -q are the same rotation.
  close(angleBetweenRotationsDegrees(identity, {x: 0, y: 0, z: 0, w: -1}), 0)
})

test('hold still: jitter is the spread around the mean, creep is first to last', () => {
  const samples = [
    sample(0, -0.01, 0, 0), sample(1, 0.01, 0, 0), sample(2, -0.01, 0, 0), sample(3, 0.01, 0, 0),
  ]
  const result = summarizeHoldStill(samples)
  close(result.positionJitterRms, 0.01)
  close(result.positionJitterMax, 0.01)
  close(result.positionCreep, 0.02)
  close(result.rotationJitterRmsDegrees, 0)
  close(result.seconds, 3)
})

test('hold still: rotation jitter is measured around the mean rotation', () => {
  const samples = [
    sample(0, 0, 0, 0, {rotation: rotationAboutY(-1)}),
    sample(1, 0, 0, 0, {rotation: rotationAboutY(1)}),
  ]
  const result = summarizeHoldStill(samples)
  close(result.rotationJitterRmsDegrees, 1, 1e-6)
  close(result.rotationCreepDegrees, 2, 1e-6)
})

test('loop: closure error is the gap between start and end as a share of the walk', () => {
  // A 1 x 1 square walked in steps of 0.25 every 0.2 s, ending 0.04 short of the start.
  const corners = [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0.04]]
  const samples = []
  let seconds = 0
  for (let side = 0; side < 4; side++) {
    const [fromX, fromZ] = corners[side]
    const [toX, toZ] = corners[side + 1]
    for (let step = side === 0 ? 0 : 1; step <= 4; step++) {
      const t = step / 4
      samples.push(sample(seconds, fromX + t * (toX - fromX), 0, fromZ + t * (toZ - fromZ)))
      seconds += 0.2
    }
  }
  const result = summarizeLoop(samples)
  close(result.pathLength, 3.96, 1e-9)
  close(result.closureError, 0.04)
  close(result.closureErrorPercentOfPath, 100 * 0.04 / 3.96)
})

test('loop: jitter between closely spaced frames is not counted as distance walked', () => {
  const samples = []
  for (let i = 0; i <= 600; i++) {
    samples.push(sample(i / 60, i % 2 ? 0.002 : -0.002, 0, 0))
  }
  // 10 s of 4 mm frame-to-frame jitter would add 2.4 m if every frame were summed.
  assert.ok(summarizeLoop(samples).pathLength < 0.5)
})

test('lens cover: counts time not tracking and the pose jump after recovery', () => {
  const samples = [
    sample(0, 0, 0, 0),
    sample(1, 0, 0, 0, {tracking: 'LIMITED'}),
    sample(3, 0.2, 0, 0, {tracking: 'LIMITED'}),
    sample(4, 0.03, 0.04, 0, {rotation: rotationAboutY(5)}),
  ]
  const result = summarizeLensCover(samples)
  close(result.secondsNotTracking, 3)
  close(result.positionJump, 0.05)
  close(result.rotationJumpDegrees, 5, 1e-6)
})

test('frames: separates display rate from tracker rate and finds stalls', () => {
  // 60 display frames a second, the tracker processing every other one, one 100 ms stall.
  const samples = []
  let seconds = 0
  for (let i = 0; i < 121; i++) {
    const trackerTimes = i % 2
      ? {trackerGpuStepMs: null, trackerCpuStepMs: null}
      : {trackerGpuStepMs: 3, trackerCpuStepMs: 5}
    samples.push(sample(seconds, 0, 0, 0, trackerTimes))
    seconds += i === 60 ? 0.1 : 1 / 60
  }
  const result = summarizeFrames(samples)
  assert.equal(result.framesOver50Ms, 1)
  close(result.frameIntervalMaxMs, 100, 1e-6)
  close(result.frameIntervalMedianMs, 1000 / 60, 1e-6)
  close(result.trackerMeanMs, 8)
  close(result.trackerGpuStepMeanMs, 3)
  close(result.trackerCpuStepMeanMs, 5)
  close(result.displayFramesPerSecond, 120 / result.seconds)
  close(result.trackerFramesPerSecond, 61 / result.seconds)
})

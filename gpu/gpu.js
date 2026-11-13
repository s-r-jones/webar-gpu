// Runs a tracker's feature front end on the live camera entirely in WebGPU compute: image
// pyramid, corner score, one best corner per grid cell, orientation, 256-bit descriptor,
// then a match of every corner against the previous frame. Only the keypoints, descriptors
// and matches come back to the CPU. The page shows what that costs and how well corners
// are followed from frame to frame.

// Pyramid step and level count match the 8th Wall engine's detector (c8/pixels/gr8-pyramid.h).
const LEVEL_STEP = 1.44
const LEVEL_COUNT = 4
const BASE_WIDTH = 480
// At most one keypoint per 16x16 cell. Corners nearer than 20 px to an edge are skipped
// so the rotated descriptor patch (radius under 19 px) stays inside the image.
const CELL = 16
const BORDER = 20
// A corner is looked for up to 13% of the picture width from where it was last frame
// (4 cells on the largest pyramid level).
const SEARCH_REACH = 0.13
// Lines on screen show one frame of motion drawn this many times longer.
const MOTION_LINE_SCALE = 8
// A match may differ in at most 64 of 256 descriptor bits and must beat the runner-up by a
// fifth; ORB-SLAM's matcher uses 50 to 100 bits and a ratio of 0.6 to 0.9.
const MATCH_BITS = 64
// Half a second at 30 camera frames a second.
const LONG_TRACK_FRAMES = 15
const KEYPOINT_BYTES = 16
const DESCRIPTOR_BYTES = 32
const TRACK_BYTES = 16

const SHARED = /* wgsl */`
struct LevelInfo { size: vec2u, cells: vec2u, slotOffset: u32, threshold: f32 }
struct Keypoint { position: vec2f, score: f32, angle: f32 }
// previousPosition: where the corner was last frame. age: frames it has been followed for.
// bitsDifferent: descriptor bits that differ from last frame.
struct Track { previousPosition: vec2f, age: u32, bitsDifferent: u32 }
const CELL = ${CELL}u;
const BORDER = ${BORDER}u;
`

// Safari hands WebGPU the camera sensor's own landscape frame even when the phone is held
// upright; Chrome hands over the upright picture. The rear sensor's frame is the upright
// picture turned a quarter turn anticlockwise (why iPhone portrait photos carry EXIF
// orientation 6), so a frame whose shape disagrees with the video's is turned back here.
const CAMERA = /* wgsl */`
@group(0) @binding(0) var camera: texture_external;
@group(0) @binding(1) var cameraSampler: sampler;

fn cameraColour(upright: vec2f, uprightSize: vec2f) -> vec4f {
  let frameSize = textureDimensions(camera);
  var inFrame = upright;
  if ((frameSize.x > frameSize.y) != (uprightSize.x > uprightSize.y)) {
    inFrame = vec2f(upright.y, 1.0 - upright.x);
  }
  return textureSampleBaseClampToEdge(camera, cameraSampler, inFrame);
}
`

const GRAY_FROM_CAMERA = /* wgsl */`
${CAMERA}
@group(0) @binding(2) var grayOut: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(grayOut);
  if (id.x >= size.x || id.y >= size.y) { return; }
  let uv = (vec2f(id.xy) + 0.5) / vec2f(size);
  let colour = cameraColour(uv, vec2f(size)).rgb;
  textureStore(grayOut, id.xy, vec4f(dot(colour, vec3f(0.299, 0.587, 0.114)), 0.0, 0.0, 1.0));
}`

// Smooths with a 3x3 kernel while shrinking, so the smaller level does not alias.
const SHRINK = /* wgsl */`
@group(0) @binding(0) var grayIn: texture_2d<f32>;
@group(0) @binding(1) var grayOut: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(grayOut);
  if (id.x >= size.x || id.y >= size.y) { return; }
  let sourceSize = vec2i(textureDimensions(grayIn));
  let centre = vec2i((vec2f(id.xy) + 0.5) * vec2f(sourceSize) / vec2f(size));
  var sum = 0.0;
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let weight = f32((2 - abs(dx)) * (2 - abs(dy)));
      let at = clamp(centre + vec2i(dx, dy), vec2i(0), sourceSize - 1);
      sum += weight * textureLoad(grayIn, at, 0).r;
    }
  }
  textureStore(grayOut, id.xy, vec4f(sum / 16.0, 0.0, 0.0, 1.0));
}`

// 5x5 binomial blur. Descriptors compare blurred pixels so single-pixel noise cannot flip bits.
const BLUR = /* wgsl */`
@group(0) @binding(0) var grayIn: texture_2d<f32>;
@group(0) @binding(1) var blurOut: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let size = vec2i(textureDimensions(blurOut));
  let at = vec2i(id.xy);
  if (at.x >= size.x || at.y >= size.y) { return; }
  var weights = array<f32, 5>(1.0, 4.0, 6.0, 4.0, 1.0);
  var sum = 0.0;
  for (var dy = -2; dy <= 2; dy++) {
    for (var dx = -2; dx <= 2; dx++) {
      let nearby = clamp(at + vec2i(dx, dy), vec2i(0), size - 1);
      sum += weights[dx + 2] * weights[dy + 2] * textureLoad(grayIn, nearby, 0).r;
    }
  }
  textureStore(blurOut, id.xy, vec4f(sum / 256.0, 0.0, 0.0, 1.0));
}`

// FAST-9 corner test: a pixel is a corner when 9 neighbouring pixels in a row on a radius-3
// circle are all brighter or all darker than it by the threshold. Score is the total contrast.
const CORNER_SCORE = /* wgsl */`
${SHARED}
@group(0) @binding(0) var grayIn: texture_2d<f32>;
@group(0) @binding(1) var scoreOut: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> level: LevelInfo;

fn hasNineInARow(mask: u32) -> bool {
  let doubled = mask | (mask << 16u);
  var run = doubled;
  for (var shift = 1u; shift <= 8u; shift++) {
    run &= doubled >> shift;
  }
  return (run & 0xffffu) != 0u;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let size = vec2i(textureDimensions(scoreOut));
  let at = vec2i(id.xy);
  if (at.x >= size.x || at.y >= size.y) { return; }
  var circle = array<vec2i, 16>(
    vec2i(0, -3), vec2i(1, -3), vec2i(2, -2), vec2i(3, -1), vec2i(3, 0), vec2i(3, 1),
    vec2i(2, 2), vec2i(1, 3), vec2i(0, 3), vec2i(-1, 3), vec2i(-2, 2), vec2i(-3, 1),
    vec2i(-3, 0), vec2i(-3, -1), vec2i(-2, -2), vec2i(-1, -3));
  var score = 0.0;
  if (at.x >= 3 && at.y >= 3 && at.x < size.x - 3 && at.y < size.y - 3) {
    let centre = textureLoad(grayIn, at, 0).r;
    var brighter = 0u;
    var darker = 0u;
    var contrast = 0.0;
    for (var i = 0u; i < 16u; i++) {
      let difference = textureLoad(grayIn, at + circle[i], 0).r - centre;
      if (difference > level.threshold) { brighter |= 1u << i; }
      if (difference < -level.threshold) { darker |= 1u << i; }
      contrast += abs(difference);
    }
    if (hasNineInARow(brighter) || hasNineInARow(darker)) { score = contrast; }
  }
  textureStore(scoreOut, id.xy, vec4f(score, 0.0, 0.0, 1.0));
}`

// One invocation per grid cell: keeps the strongest corner in the cell. This fixes the
// output size and spreads keypoints across the image without any sorting on the CPU.
const SELECT = /* wgsl */`
${SHARED}
@group(0) @binding(0) var scoreIn: texture_2d<f32>;
@group(0) @binding(1) var<uniform> level: LevelInfo;
@group(0) @binding(2) var<storage, read_write> keypoints: array<Keypoint>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= level.cells.x || id.y >= level.cells.y) { return; }
  let origin = vec2i(id.xy * CELL + BORDER);
  var best = 0.0;
  var bestAt = origin;
  for (var y = 0; y < i32(CELL); y++) {
    for (var x = 0; x < i32(CELL); x++) {
      let score = textureLoad(scoreIn, origin + vec2i(x, y), 0).r;
      if (score > best) {
        best = score;
        bestAt = origin + vec2i(x, y);
      }
    }
  }
  let slot = level.slotOffset + id.y * level.cells.x + id.x;
  keypoints[slot] = Keypoint(vec2f(bestAt) + 0.5, best, 0.0);
}`

// One invocation per keypoint: orientation from the intensity centroid of a radius-15 patch
// (as in ORB), then 256 brightness comparisons at pattern offsets rotated by that angle.
const DESCRIBE = /* wgsl */`
${SHARED}
@group(0) @binding(0) var blurIn: texture_2d<f32>;
@group(0) @binding(1) var<uniform> level: LevelInfo;
@group(0) @binding(2) var<storage, read_write> keypoints: array<Keypoint>;
@group(0) @binding(3) var<storage, read_write> descriptors: array<array<u32, 8>>;
@group(0) @binding(4) var<storage, read> comparisonPattern: array<vec4i, 256>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= level.cells.x * level.cells.y) { return; }
  let slot = level.slotOffset + id.x;
  let keypoint = keypoints[slot];
  if (keypoint.score == 0.0) { return; }
  let centre = vec2i(keypoint.position);

  var momentX = 0.0;
  var momentY = 0.0;
  for (var y = -15; y <= 15; y++) {
    for (var x = -15; x <= 15; x++) {
      if (x * x + y * y <= 225) {
        let value = textureLoad(blurIn, centre + vec2i(x, y), 0).r;
        momentX += f32(x) * value;
        momentY += f32(y) * value;
      }
    }
  }
  let angle = atan2(momentY, momentX);
  let cosine = cos(angle);
  let sine = sin(angle);

  var bits = array<u32, 8>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
  for (var i = 0u; i < 256u; i++) {
    let offsets = vec4f(comparisonPattern[i]);
    let first = vec2i(round(vec2f(cosine * offsets.x - sine * offsets.y, sine * offsets.x + cosine * offsets.y)));
    let second = vec2i(round(vec2f(cosine * offsets.z - sine * offsets.w, sine * offsets.z + cosine * offsets.w)));
    if (textureLoad(blurIn, centre + first, 0).r < textureLoad(blurIn, centre + second, 0).r) {
      bits[i / 32u] |= 1u << (i % 32u);
    }
  }
  descriptors[slot] = bits;
  keypoints[slot].angle = angle;
}`

// One invocation per keypoint: finds the same corner in the previous frame by comparing
// descriptors with the keypoints of nearby cells on the same pyramid level.
const FOLLOW = /* wgsl */`
${SHARED}
@group(0) @binding(0) var<uniform> level: LevelInfo;
@group(0) @binding(1) var<storage, read> keypoints: array<Keypoint>;
@group(0) @binding(2) var<storage, read> descriptors: array<array<u32, 8>>;
@group(0) @binding(3) var<storage, read> previousKeypoints: array<Keypoint>;
@group(0) @binding(4) var<storage, read> previousDescriptors: array<array<u32, 8>>;
@group(0) @binding(5) var<storage, read> previousTracks: array<Track>;
@group(0) @binding(6) var<storage, read_write> tracks: array<Track>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= level.cells.x * level.cells.y) { return; }
  let slot = level.slotOffset + id.x;
  let keypoint = keypoints[slot];
  var track = Track(keypoint.position, 0u, 256u);
  if (keypoint.score > 0.0) {
    let cells = vec2i(level.cells);
    let cell = vec2i(i32(id.x) % cells.x, i32(id.x) / cells.x);
    let reach = i32(ceil(f32(level.size.x) * ${SEARCH_REACH} / f32(CELL)));
    var best = 256u;
    var runnerUp = 256u;
    var bestSlot = 0u;
    var bestPosition = vec2f(-1000.0, -1000.0);
    for (var dy = -reach; dy <= reach; dy++) {
      for (var dx = -reach; dx <= reach; dx++) {
        let nearby = cell + vec2i(dx, dy);
        if (nearby.x < 0 || nearby.y < 0 || nearby.x >= cells.x || nearby.y >= cells.y) { continue; }
        let nearbySlot = level.slotOffset + u32(nearby.y * cells.x + nearby.x);
        let candidate = previousKeypoints[nearbySlot];
        if (candidate.score == 0.0) { continue; }
        var bitsDifferent = 0u;
        for (var word = 0u; word < 8u; word++) {
          bitsDifferent += countOneBits(descriptors[slot][word] ^ previousDescriptors[nearbySlot][word]);
        }
        // Two neighbouring cells can each pick a pixel of the same corner. Those are one
        // candidate, not a rival, so they do not count as the runner-up.
        let sameCorner = distance(candidate.position, bestPosition) < 6.0;
        if (bitsDifferent < best) {
          if (!sameCorner) { runnerUp = best; }
          best = bitsDifferent;
          bestSlot = nearbySlot;
          bestPosition = candidate.position;
        } else if (!sameCorner && bitsDifferent < runnerUp) {
          runnerUp = bitsDifferent;
        }
      }
    }
    if (best <= ${MATCH_BITS}u && best * 10u < runnerUp * 8u) {
      track = Track(bestPosition, previousTracks[bestSlot].age + 1u, best);
    }
  }
  tracks[slot] = track;
}`

const DRAW = /* wgsl */`
${SHARED}
// cover: how much of the camera picture fits the screen on each axis (the rest is cropped).
// levels: for each pyramid level, its width, height and first keypoint slot.
struct View { cover: vec2f, pointSize: vec2f, levels: array<vec4f, ${LEVEL_COUNT}> }

${CAMERA}
@group(0) @binding(2) var<uniform> view: View;
@group(0) @binding(3) var<storage, read> keypoints: array<Keypoint>;
@group(0) @binding(4) var<storage, read> tracks: array<Track>;

struct Corner { @builtin(position) position: vec4f, @location(0) uv: vec2f }

@vertex
fn cameraVertex(@builtin(vertex_index) index: u32) -> Corner {
  var corners = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  let corner = corners[index];
  return Corner(vec4f(corner, 0.0, 1.0), vec2f(corner.x * 0.5 + 0.5, 0.5 - corner.y * 0.5));
}

@fragment
fn cameraFragment(in: Corner) -> @location(0) vec4f {
  return cameraColour((in.uv - 0.5) * view.cover + 0.5, view.levels[0].xy);
}

struct Dot { @builtin(position) position: vec4f, @location(0) colour: vec3f }
const OFF_SCREEN = vec4f(2.0, 2.0, 2.0, 1.0);

fn levelOf(slot: u32) -> u32 {
  var levelIndex = 0u;
  for (var i = 1u; i < ${LEVEL_COUNT}u; i++) {
    if (f32(slot) >= view.levels[i].z) { levelIndex = i; }
  }
  return levelIndex;
}

fn onScreen(inLevel: vec2f, levelIndex: u32) -> vec2f {
  let inPicture = inLevel / view.levels[levelIndex].xy;
  let covered = (inPicture - 0.5) / view.cover + 0.5;
  return vec2f(covered.x * 2.0 - 1.0, 1.0 - covered.y * 2.0);
}

// Red: new this frame. Yellow: followed for under half a second. Green: half a second or more.
fn ageColour(age: u32) -> vec3f {
  if (age >= ${LONG_TRACK_FRAMES}u) { return vec3f(0.2, 1.0, 0.4); }
  if (age > 0u) { return vec3f(1.0, 0.9, 0.1); }
  return vec3f(1.0, 0.25, 0.3);
}

@vertex
fn keypointVertex(@builtin(vertex_index) index: u32, @builtin(instance_index) slot: u32) -> Dot {
  let keypoint = keypoints[slot];
  if (keypoint.score == 0.0) { return Dot(OFF_SCREEN, vec3f(0.0)); }
  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0));
  let levelIndex = levelOf(slot);
  let size = view.pointSize * (1.0 + 0.5 * f32(levelIndex));
  let centre = onScreen(keypoint.position, levelIndex);
  return Dot(vec4f(centre + corners[index] * size, 0.0, 1.0), ageColour(tracks[slot].age));
}

// A thin bar behind each followed corner: its motion since last frame, drawn longer.
@vertex
fn trackVertex(@builtin(vertex_index) index: u32, @builtin(instance_index) slot: u32) -> Dot {
  let keypoint = keypoints[slot];
  let track = tracks[slot];
  let levelIndex = levelOf(slot);
  let motion = keypoint.position - track.previousPosition;
  let lineStart = onScreen(keypoint.position - motion * ${MOTION_LINE_SCALE}.0, levelIndex);
  let lineEnd = onScreen(keypoint.position, levelIndex);
  // Measured in dot sizes so the bar is equally thick in every direction.
  let along = (lineEnd - lineStart) / view.pointSize;
  if (keypoint.score == 0.0 || track.age == 0u || length(along) < 0.5) { return Dot(OFF_SCREEN, vec3f(0.0)); }
  let across = normalize(vec2f(-along.y, along.x)) * view.pointSize * 0.3;
  var ends = array<f32, 6>(0.0, 1.0, 0.0, 0.0, 1.0, 1.0);
  var sides = array<f32, 6>(-1.0, -1.0, 1.0, 1.0, -1.0, 1.0);
  return Dot(vec4f(mix(lineStart, lineEnd, ends[index]) + across * sides[index], 0.0, 1.0), ageColour(track.age));
}

@fragment
fn keypointFragment(in: Dot) -> @location(0) vec4f {
  return vec4f(in.colour, 1.0);
}`

// 256 pairs of offsets, each coordinate drawn from a normal distribution (sigma 5 px,
// clipped to 13 px) as in the original BRIEF descriptor. A fixed seed keeps descriptors
// comparable between frames and sessions.
function comparisonPattern() {
  let state = 12345
  const uniform = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0
    return (state + 1) / 4294967297
  }
  const normal = () => Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform())
  return Int32Array.from({length: 256 * 4}, () => Math.max(-13, Math.min(13, Math.round(5 * normal()))))
}

function pyramidLevels(videoWidth, videoHeight) {
  const levels = []
  let slotOffset = 0
  for (let i = 0; i < LEVEL_COUNT; i++) {
    const width = Math.round(BASE_WIDTH / LEVEL_STEP ** i)
    const height = Math.round(width * videoHeight / videoWidth)
    const cellsX = Math.floor((width - 2 * BORDER) / CELL)
    const cellsY = Math.floor((height - 2 * BORDER) / CELL)
    levels.push({width, height, cellsX, cellsY, slotOffset})
    slotOffset += cellsX * cellsY
  }
  return {levels, slotCount: slotOffset}
}

export async function startFeatureTest({canvas, video, onStats, onError}) {
  const adapter = await navigator.gpu.requestAdapter()
  const canTimeGpu = adapter.features.has('timestamp-query')
  const device = await adapter.requestDevice({requiredFeatures: canTimeGpu ? ['timestamp-query'] : []})
  device.addEventListener('uncapturederror', event => onError(event.error.message))

  const context = canvas.getContext('webgpu')
  const format = navigator.gpu.getPreferredCanvasFormat()
  context.configure({device, format, alphaMode: 'opaque'})

  const {levels, slotCount} = pyramidLevels(video.videoWidth, video.videoHeight)
  const texture = ({width, height}) => device.createTexture({
    size: [width, height],
    format: 'r32float',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  }).createView()
  for (const level of levels) {
    level.gray = texture(level)
    level.blur = texture(level)
    level.score = texture(level)
    level.info = device.createBuffer({size: 24, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST})
    const info = new DataView(new ArrayBuffer(24))
    ;[level.width, level.height, level.cellsX, level.cellsY, level.slotOffset].forEach(
      (value, i) => info.setUint32(4 * i, value, true))
    // A corner must differ from its ring by 6% of full brightness (about 15 of 255).
    info.setFloat32(20, 0.06, true)
    device.queue.writeBuffer(level.info, 0, info)
  }

  const perSlotBuffer = bytesPerSlot => device.createBuffer({
    size: slotCount * bytesPerSlot,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  })
  const keypoints = perSlotBuffer(KEYPOINT_BYTES)
  const descriptors = perSlotBuffer(DESCRIPTOR_BYTES)
  const tracks = perSlotBuffer(TRACK_BYTES)
  const results = [keypoints, descriptors, tracks]
  // Last frame's results, which this frame's corners are matched against.
  const previousResults = [perSlotBuffer(KEYPOINT_BYTES), perSlotBuffer(DESCRIPTOR_BYTES), perSlotBuffer(TRACK_BYTES)]
  const pattern = device.createBuffer({size: 256 * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST})
  device.queue.writeBuffer(pattern, 0, comparisonPattern())

  const resultBytes = slotCount * (KEYPOINT_BYTES + DESCRIPTOR_BYTES + TRACK_BYTES)
  // Results are read back a frame or two after they are produced, so a few buffers rotate.
  const freeReadbacks = [0, 1, 2].map(() => device.createBuffer({
    size: resultBytes + 16,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  }))
  const timing = canTimeGpu && {
    queries: device.createQuerySet({type: 'timestamp', count: 2}),
    resolved: device.createBuffer({size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC}),
  }

  const computePipeline = code => device.createComputePipeline({
    layout: 'auto',
    compute: {module: device.createShaderModule({code}), entryPoint: 'main'},
  })
  const pipelines = {
    grayFromCamera: computePipeline(GRAY_FROM_CAMERA),
    shrink: computePipeline(SHRINK),
    blur: computePipeline(BLUR),
    cornerScore: computePipeline(CORNER_SCORE),
    select: computePipeline(SELECT),
    describe: computePipeline(DESCRIBE),
    follow: computePipeline(FOLLOW),
  }
  const bind = (pipeline, resources) => device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: resources.map((resource, binding) => ({binding, resource})),
  })
  const sampler = device.createSampler({magFilter: 'linear', minFilter: 'linear'})
  // Every pass except the first reads only textures this page owns, so its bindings never change.
  for (const [i, level] of levels.entries()) {
    level.passes = [
      i > 0 && [pipelines.shrink, bind(pipelines.shrink, [levels[i - 1].gray, level.gray]), level.width, level.height, 8],
      [pipelines.blur, bind(pipelines.blur, [level.gray, level.blur]), level.width, level.height, 8],
      [pipelines.cornerScore, bind(pipelines.cornerScore, [level.gray, level.score, {buffer: level.info}]), level.width, level.height, 8],
      [pipelines.select, bind(pipelines.select, [level.score, {buffer: level.info}, {buffer: keypoints}]), level.cellsX, level.cellsY, 8],
      [pipelines.describe, bind(pipelines.describe, [level.blur, {buffer: level.info}, {buffer: keypoints}, {buffer: descriptors}, {buffer: pattern}]), level.cellsX * level.cellsY, 1, 64],
      [pipelines.follow, bind(pipelines.follow, [{buffer: level.info}, ...[...results.slice(0, 2), ...previousResults, tracks].map(buffer => ({buffer}))]), level.cellsX * level.cellsY, 1, 64],
    ].filter(Boolean)
  }

  const drawModule = device.createShaderModule({code: DRAW})
  const drawPipeline = (vertex, fragment) => device.createRenderPipeline({
    layout: 'auto',
    vertex: {module: drawModule, entryPoint: vertex},
    fragment: {module: drawModule, entryPoint: fragment, targets: [{format}]},
  })
  const cameraPipeline = drawPipeline('cameraVertex', 'cameraFragment')
  const keypointPipeline = drawPipeline('keypointVertex', 'keypointFragment')
  const trackPipeline = drawPipeline('trackVertex', 'keypointFragment')
  const view = device.createBuffer({size: 16 + 16 * LEVEL_COUNT, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST})

  function writeView() {
    canvas.width = canvas.clientWidth * Math.min(2, window.devicePixelRatio)
    canvas.height = canvas.clientHeight * Math.min(2, window.devicePixelRatio)
    const screenShape = canvas.width / canvas.height
    const pictureShape = video.videoWidth / video.videoHeight
    const cover = screenShape > pictureShape ? [1, pictureShape / screenShape] : [screenShape / pictureShape, 1]
    const dot = 3 * Math.min(2, window.devicePixelRatio)
    device.queue.writeBuffer(view, 0, new Float32Array([
      ...cover, dot / canvas.width, dot / canvas.height,
      ...levels.flatMap(level => [level.width, level.height, level.slotOffset, 0]),
    ]))
  }
  writeView()
  window.addEventListener('resize', writeView)
  const keypointBindings = pipeline => device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{binding: 2, resource: {buffer: view}}, {binding: 3, resource: {buffer: keypoints}}, {binding: 4, resource: {buffer: tracks}}],
  })
  const overlays = [trackPipeline, keypointPipeline].map(pipeline => [pipeline, keypointBindings(pipeline)])

  function frame() {
    const startedAt = performance.now()
    const cameraFrame = device.importExternalTexture({source: video})
    const encoder = device.createCommandEncoder()

    const compute = encoder.beginComputePass(timing ? {
      timestampWrites: {querySet: timing.queries, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1},
    } : {})
    compute.setPipeline(pipelines.grayFromCamera)
    compute.setBindGroup(0, bind(pipelines.grayFromCamera, [cameraFrame, sampler, levels[0].gray]))
    compute.dispatchWorkgroups(Math.ceil(levels[0].width / 8), Math.ceil(levels[0].height / 8))
    for (const level of levels) {
      for (const [pipeline, bindGroup, countX, countY, groupSize] of level.passes) {
        compute.setPipeline(pipeline)
        compute.setBindGroup(0, bindGroup)
        compute.dispatchWorkgroups(Math.ceil(countX / groupSize), countY === 1 ? 1 : Math.ceil(countY / groupSize))
      }
    }
    compute.end()

    const draw = encoder.beginRenderPass({
      colorAttachments: [{view: context.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store'}],
    })
    draw.setPipeline(cameraPipeline)
    draw.setBindGroup(0, device.createBindGroup({
      layout: cameraPipeline.getBindGroupLayout(0),
      entries: [{binding: 0, resource: cameraFrame}, {binding: 1, resource: sampler}, {binding: 2, resource: {buffer: view}}],
    }))
    draw.draw(3)
    for (const [pipeline, bindings] of overlays) {
      draw.setPipeline(pipeline)
      draw.setBindGroup(0, bindings)
      draw.draw(6, slotCount)
    }
    draw.end()

    const readback = freeReadbacks.pop()
    if (readback) {
      let offset = 0
      for (const result of results) {
        encoder.copyBufferToBuffer(result, 0, readback, offset, result.size)
        offset += result.size
      }
      if (timing) {
        encoder.resolveQuerySet(timing.queries, 0, 2, timing.resolved, 0)
        encoder.copyBufferToBuffer(timing.resolved, 0, readback, resultBytes, 16)
      }
    }
    for (const [i, result] of results.entries()) {
      encoder.copyBufferToBuffer(result, 0, previousResults[i], 0, result.size)
    }
    device.queue.submit([encoder.finish()])
    const mainThreadMs = performance.now() - startedAt

    if (readback) {
      readback.mapAsync(GPUMapMode.READ).then(() => {
        const bytes = readback.getMappedRange()
        const scores = new Float32Array(bytes, 0, slotCount * 4)
        const trackFields = new Uint32Array(bytes, slotCount * (KEYPOINT_BYTES + DESCRIPTOR_BYTES), slotCount * 4)
        let keypointCount = 0
        let matchedCount = 0
        let longTrackCount = 0
        let bitsDifferentTotal = 0
        for (let slot = 0; slot < slotCount; slot++) {
          if (scores[4 * slot + 2] > 0) {
            keypointCount++
            const age = trackFields[4 * slot + 2]
            if (age > 0) {
              matchedCount++
              bitsDifferentTotal += trackFields[4 * slot + 3]
            }
            if (age >= LONG_TRACK_FRAMES) {
              longTrackCount++
            }
          }
        }
        const [began, ended] = timing ? new BigUint64Array(bytes, resultBytes, 2) : []
        onStats({
          keypointCount,
          matchedCount,
          longTrackCount,
          bitsDifferentPerMatch: matchedCount ? bitsDifferentTotal / matchedCount : 0,
          mainThreadMs,
          gpuMs: timing ? Number(ended - began) / 1e6 : null,
          resultsAfterMs: performance.now() - startedAt,
          readbackKilobytes: resultBytes / 1024,
        })
        readback.unmap()
        freeReadbacks.push(readback)
      })
    }
    nextFrame()
  }

  // Runs once per new camera frame where the browser can say so, else once per display frame.
  const nextFrame = () => (video.requestVideoFrameCallback
    ? video.requestVideoFrameCallback(frame)
    : requestAnimationFrame(frame))
  nextFrame()
  return {levels, slotCount, canTimeGpu}
}

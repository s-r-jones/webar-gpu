// Hooks that must be in place before the engine script loads.

// Keeps a handle on every WebAssembly memory the engine creates so the page can report
// how much the tracker has allocated and change its settings.
window.capturedWasmMemories = []

for (const name of ['instantiate', 'instantiateStreaming']) {
  const original = WebAssembly[name]
  WebAssembly[name] = async (...args) => {
    const result = await original.apply(WebAssembly, args)
    // instantiate(bytes) resolves to {module, instance}; instantiate(module) to the instance.
    const instance = result.instance || result
    if (instance.exports.memory) {
      window.capturedWasmMemories.push(instance.exports.memory)
    }
    return result
  }
}

// The engine prints the reason for an abort to the console just before it stops; the last
// few lines go into the crash report.
window.engineMessages = []
for (const level of ['error', 'warn']) {
  const original = console[level]
  console[level] = (...args) => {
    window.engineMessages.push(args.join(' '))
    if (window.engineMessages.length > 20) {
      window.engineMessages.shift()
    }
    original.apply(console, args)
  }
}

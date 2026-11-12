// Keeps a handle on every WebAssembly memory the engine creates so the page can report
// how much the tracker has allocated. Must load before the engine script.
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

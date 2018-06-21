function downloadText(text) {
  let dw = document.createElement('a');
  dw.setAttribute('href', 'data:text/plain,' + encodeURIComponent(text));
  dw.setAttribute('download', 'download.txt');
  dw.style.display = 'none';
  document.body.appendChild(dw);
  dw.click();
  document.body.removeChild(dw);
}

function getVariableValues() {
  return {
    G1w: { shape: G1w.shape, data: Array.from(G1w.dataSync()) },
    G1b: { shape: G1b.shape, data: Array.from(G1b.dataSync()) },
    G2w: { shape: G2w.shape, data: Array.from(G2w.dataSync()) },
    G2b: { shape: G2b.shape, data: Array.from(G2b.dataSync()) },
    G3w: { shape: G3w.shape, data: Array.from(G3w.dataSync()) },
    G3b: { shape: G3b.shape, data: Array.from(G3b.dataSync()) },
    D1w: { shape: D1w.shape, data: Array.from(D1w.dataSync()) },
    D1b: { shape: D1b.shape, data: Array.from(D1b.dataSync()) },
    D2w: { shape: D2w.shape, data: Array.from(D2w.dataSync()) },
    D2b: { shape: D2b.shape, data: Array.from(D2b.dataSync()) },
    D3w: { shape: D3w.shape, data: Array.from(D3w.dataSync()) },
    D3b: { shape: D3b.shape, data: Array.from(D3b.dataSync()) }
  }
}


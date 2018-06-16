class ImageUtil {

  /**
   * Flattens a RGBA channel data into grey scale float array
   */
  static flatten(data, options) {
    const w = options.width || 0;
    const h = options.height || 0;

    const flat = [];

    for (let i = 0; i < w * h; ++i) {
      const j = i * 4;
      const newVal = (data[j+0] + data[j+1] + data[j+2] + data[j+3]) / 4.0 ;
      flat.push( newVal / 255.0 );
    }
    return flat;
  }

  /**
   * Unflatten single channel to RGBA
   */
  static unflatten(data, options) {
    const w = options.width || 0;
    const h = options.height || 0;
    const unflat = [];
    for (let i = 0; i < w * h; ++i) {
      const val = data[i];
      unflat.push(data[i] * 255);
      unflat.push(data[i] * 255);
      unflat.push(data[i] * 255);
      unflat.push(255);
    }
    return unflat;
  }


  /**
   * Load image data into RGBA numeric array, which is
   * somewhat compatible with tensor data structure
   *
   * @param {string} url - the image URL
   * @param {object} options
   * @param {number} options.width - optional, resize to specified width
   * @param {number} options.height - optional, resize to specified height 
   *
   * FIXME: reuse canvas element if available
   * FIXME: add option for channel filters
   */
  static async loadImage(url, options = {}) {
    const img = new Image();
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    window.ctx = ctx;

    const imgRequest = new Promise((resolve, reject) => {
      img.crossOrigin = '';
      img.onload = () => {
        img.width = options.width || img.naturalWidth;
        img.height = options.height || img.naturalHeight;

        canvas.width = img.width;
        canvas.height = img.height;

        ctx.drawImage(img, 0 , 0, img.width, img.height);
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        // ctx.drawImage(img, 0, i * chunkSize, img.width, chunkSize, 0, 0, img.width, chunkSize);

        resolve(imageData);
      };
      img.src = url;
    });
    return imgRequest;
  }
}

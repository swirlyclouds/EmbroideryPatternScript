
// Get Pixel Data
function getPixel(url, x, y) {
    var img = new Image();
    img.src = url;
    var canvas = document.createElement('canvas');
    var context = canvas.getContext('2d');
    context.drawImage(img, 0, 0);
    return context.getImageData(x, y, 1, 1).data;
  }
  getPixel('./bg.png', 10, 10); // [255, 255, 255, 0];
  
  // https://www.w3schools.com/jsref/canvas_getimagedata.asp


// RGB to LAB
  // https://github.com/antimatter15/rgb-lab/blob/master/color.js

// k-Means Clustering
  // https://medium.com/geekculture/implementing-k-means-clustering-from-scratch-in-javascript-13d71fbcb31e 



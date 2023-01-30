let videoWidth, videoHeight;

let streaming = false;

let video = document.getElementById("video");
let canvasOutput = document.getElementById("canvasOutput");
let canvasOutputCtx = canvasOutput.getContext("2d");
let stream = null;

let detectFace = document.getElementById("face");
let detectEye = document.getElementById("eye");

function startCamera() {
  if (streaming) return;
  navigator.mediaDevices
    .getUserMedia({ video: true, audio: false })
    .then(function (s) {
      stream = s;
      video.srcObject = s;
      video.play();
    })
    .catch(function (err) {
      console.log("An error occured! " + err);
    });

  video.addEventListener(
    "canplay",
    function (ev) {
      if (!streaming) {
        videoWidth = video.videoWidth;
        videoHeight = video.videoHeight;
        video.setAttribute("width", videoWidth);
        video.setAttribute("height", videoHeight);
        canvasOutput.width = videoWidth;
        canvasOutput.height = videoHeight;
        streaming = true;
      }
      startVideoProcessing();
    },
    false
  );
}

// const modelParams = {
//   flipHorizontal: true,   
//   maxNumBoxes: 1,        
//   iouThreshold: 0.5,     
//   scoreThreshold: 0.6,    
// }


// function midpoint(ptA, ptB) {
//   return (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5;
// }

let faceClassifier = null;
let eyeClassifier = null;

let src = null;
let dstC1 = null;
let dstC3 = null;
let dstC4 = null;

let canvasInput = null;
let canvasInputCtx = null;

let canvasBuffer = null;
let canvasBufferCtx = null;

let pixelsPerMetric = null;

function startVideoProcessing() {
  if (!streaming) {
    console.warn("Please startup your webcam");
    return;
  }
  stopVideoProcessing();
  canvasInput = document.createElement("canvas");
  canvasInput.width = videoWidth;
  canvasInput.height = videoHeight;
  canvasInputCtx = canvasInput.getContext("2d");

  canvasBuffer = document.createElement("canvas");
  canvasBuffer.width = videoWidth;
  canvasBuffer.height = videoHeight;
  canvasBufferCtx = canvasBuffer.getContext("2d");

  srcMat = new cv.Mat(videoHeight, videoWidth, cv.CV_8UC4);
  grayMat = new cv.Mat(videoHeight, videoWidth, cv.CV_8UC1);

  faceClassifier = new cv.CascadeClassifier();
  faceClassifier.load("nails2.xml");

  eyeClassifier = new cv.CascadeClassifier();
  eyeClassifier.load("haarcascade_eye.xml");

  requestAnimationFrame(processVideo);
}



function processVideo() {
  stats.begin();
  canvasInputCtx.drawImage(video, 0, 0, videoWidth, videoHeight);
  let imageData = canvasInputCtx.getImageData(0, 0, videoWidth, videoHeight);
  srcMat.data.set(imageData.data);

  cv.cvtColor(srcMat, grayMat, cv.COLOR_RGBA2GRAY);
  let faces = [];
  let eyes = [];
  let size;
  if (detectFace.checked) {
    let faceVect = new cv.RectVector();
    let faceMat = new cv.Mat();
    if (detectEye.checked) {
      cv.pyrDown(grayMat, faceMat);
      size = faceMat.size();
    } else {
      cv.pyrDown(grayMat, faceMat);
      cv.pyrDown(faceMat, faceMat);
      size = faceMat.size();
    }
    //faceClassifier.detectMultiScale(faceMat, faceVect, 5, 7); //Will work on 10000 nails dataset
    //faceClassifier.detectMultiScale(faceMat, faceVect, 10,12); //will work on 2000 nails dataset
    faceClassifier.detectMultiScale(faceMat, faceVect);
    for (let i = 0; i < faceVect.size(); i++) {
      let face = faceVect.get(i);

      faces.push(new cv.Rect(face.x, face.y, face.width, face.height));

      // if (detectEye.checked) {
      //   let eyeVect = new cv.RectVector();
      //   let eyeMat = faceMat.getRoiRect(face);
      //   eyeClassifier.detectMultiScale(eyeMat, eyeVect);
      //   for (let i = 0; i < eyeVect.size(); i++) {
      //     let eye = eyeVect.get(i);
      //     eyes.push(new cv.Rect(face.x + eye.x, face.y + eye.y, eye.width, eye.height));
      //   }
      //   eyeMat.delete();
      //   eyeVect.delete();
      // }
    }
    faceMat.delete();
    faceVect.delete();
  } else {
    if (detectEye.checked) {
      let eyeVect = new cv.RectVector();
      let eyeMat = new cv.Mat();
      cv.pyrDown(grayMat, eyeMat);
      size = eyeMat.size();
      eyeClassifier.detectMultiScale(eyeMat, eyeVect);
      for (let i = 0; i < eyeVect.size(); i++) {
        let eye = eyeVect.get(i);
        eyes.push(new cv.boundingRect(eye.x, eye.y, eye.width, eye.height));
      }
      eyeMat.delete();
      eyeVect.delete();
    }
  }
  canvasOutputCtx.drawImage(canvasInput, 0, 0, videoWidth, videoHeight);
  drawResults(canvasOutputCtx, faces, "green", size);
  drawResults(canvasOutputCtx, eyes, "yellow", size);

  stats.end();
  requestAnimationFrame(processVideo);
}

function drawResults(ctx, results, color, size, distance=14) {

  // let img;
  // const nails = ["nail1.png", "nail2.png", "nail3.png", "nail-15-1.png", "nail5.png"];
  // if (distance >= 11 && distance <= 15) {
  //     img = new Image();
  //     img.src = nails[distance - 11];
  //     ctx.fillText(`Nail ${distance - 10}`, 80, 50);
  //     ctx.font = "20px Georgia";
  // }

  for (let i = 0; i < results.length; ++i) {
    let rect = results[i];
    let xRatio = videoWidth / size.width;
    let yRatio = videoHeight / size.height;
    
    //ctx.drawImage(img, rect.x * xRatio, rect.y * yRatio, rect.width * xRatio, rect.height * yRatio);
    let width = rect.width * xRatio;
    let height = rect.height * yRatio;

    ctx.lineWidth = 2;
    ctx.strokeRect(rect.x * xRatio, rect.y * yRatio, width, height);

    let widthInMm = (rect.width * distance * 10) / size.width;
    let heightInMm = (rect.height * distance * 10) / size.height;

    ctx.font = "16px sans-serif";
    ctx.fillStyle = "white";
    ctx.textAlign = "center";
    //ctx.fillText(`Width: ${widthInMm.toFixed(1)} mm Length: ${heightInMm.toFixed(1)} mm`,rect.x * xRatio + (rect.width * xRatio) / 2, rect.y * yRatio - 10, 80, 50);
    ctx.fillText(
      `Width: ${widthInMm.toFixed(1)} mm Length: ${heightInMm.toFixed(1)} mm`,
      rect.x * xRatio + (rect.width * xRatio) / 2,
      rect.y * yRatio - 10
    );

    // console.log(
    //   `Width: ${widthInMm.toFixed(2)} mm Length: ${heightInMm.toFixed(2)} mm`
    // );
    console.log("https://cvs-test.com/exp-features Fixing This link")
  } 
}

// function drawResults(ctx, results, color, size, distance=8) {

//   // Add a variable to represent the desired rectangle size
//   let rectangleSize = 1.0;

//   for (let i = 0; i < results.length; ++i) {
//     let rect = results[i];
//     let xRatio = videoWidth / size.width;
//     let yRatio = videoHeight / size.height;

//     // Update the width and height of the rectangle using the rectangleSize variable
//     let width = rect.width * xRatio * rectangleSize;
//     let height = rect.height * yRatio * rectangleSize;

//     ctx.lineWidth = 2;
//     ctx.strokeRect(rect.x * xRatio, rect.y * yRatio, width, height);

//     let widthInMm = (rect.width * distance * 10) / size.width;
//     let heightInMm = (rect.height * distance * 10) / size.height;

//     ctx.font = "16px sans-serif";
//     ctx.fillStyle = "white";
//     ctx.textAlign = "center";
//     // ctx.fillText(
//     //   `Width: ${widthInMm.toFixed(1)} mm Length: ${heightInMm.toFixed(1)} mm`,
//     //   rect.x * xRatio + (rect.width * xRatio) / 2,
//     //   rect.y * yRatio - 10
//     // );

//     // console.log(
//     //   `Width: ${widthInMm.toFixed(2)} mm Length: ${heightInMm.toFixed(2)} mm`
//     // );
//   } 
// }

function stopVideoProcessing() {
  if (src != null && !src.isDeleted()) src.delete();
  if (dstC1 != null && !dstC1.isDeleted()) dstC1.delete();
  if (dstC3 != null && !dstC3.isDeleted()) dstC3.delete();
  if (dstC4 != null && !dstC4.isDeleted()) dstC4.delete();
}

function stopCamera() {
  if (!streaming) return;
  stopVideoProcessing();
  document
    .getElementById("canvasOutput")
    .getContext("2d")
    .clearRect(0, 0, width, height);
  video.pause();
  video.srcObject = null;
  stream.getVideoTracks()[0].stop();
  streaming = false;
}

function initUI() {
  stats = new Stats();
  stats.showPanel(0);
  document.getElementById("container").appendChild(stats.dom);
}

function opencvIsReady() {
  console.log("OpenCV.js is ready");
  initUI();
  startCamera();
  
}

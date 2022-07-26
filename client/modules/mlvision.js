import * as destructor from "./destructor.js";

const quando = window['quando']
const document = window['document']
const MOBILE_NET_INPUT_WIDTH = 224;
const MOBILE_NET_INPUT_HEIGHT = 224;
const STOP_DATA_GATHER = -1;


  if (!quando) {
    alert('Fatal Error: ML_vision must be included after quando_browser')
  }

  let self = quando.mlvision = {}

  self.trainDataInputs = [];    // set of training input data for machine learning model
  self.trainDataOutputs = [];
  self.examplesCount = [];
  self.classNames = [];
  self.classButtons = [];
  self.predict = false;
  self.mobilenet = undefined;
  self.classCounts = 0;

  self.width = window.innerWidth
  self.height = window.innerHeight

  function _enableCam() {
    alert("added");
    const VIDEO = document.getElementById('webcam');
    const ENABLE_CAM_BUTTON = document.getElementById('enableCam');
    console.log("get video and button");
    if (_hasGetUserMedia()) {
        // getUsermedia parameters.
        const constraints = {
            video: true,
            width: 640,
            height: 480
        };

        // Activate the webcam stream.
        navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
            VIDEO.srcObject = stream;
            VIDEO.addEventListener('loadeddata', function() {
                videoPlaying = true;
                ENABLE_CAM_BUTTON.classList.add('removed');
            });
        });
    } else {
        console.warn('getUserMedia() is not supported by your browser');
    }
  }

  function _hasGetUserMedia() {
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
  }
  
  self.buildModel = () => {
    const scene = self.getScene();
  }

  self.addClass = (class_name) => {
    self.classNames.push(class_name);
    let class_button = document.createElement("button");
    class_button.className = 'dataCollector';
    class_button.setAttribute("data-1hot", self.classCounts);
    class_button.setAttribute("data-name", class_name);
    class_button.innerHTML = "Gather " + class_name + " image data";
    class_button.addEventListener('mousedown', _gatherDataForClass);
    class_button.addEventListener('mouseup', _gatherDataForClass);
    self.classButtons.push(class_button);
    self.classCounts += 1;
  }

  self.getScene = () => {
    let scene = document.getElementById('quando_mlvision_js_div');
    if (scene == null) {
        scene = document.createElement('div');
        scene.setAttribute('id', 'quando_mlvision_js_div');

        // set status
        let p_status = document.createElement('p');
        p_status.setAttribute('id', 'model_status');
        p_status.innerHTML = 'awaiting model lode';
        scene.appendChild(p_status);

        // set webCam
        let camera = document.createElement('video')
        camera.setAttribute('id', 'webcam')
        camera.style.clear = 'both';
        camera.style.display = 'block';
        camera.style.margin = '10px';
        camera.style.width = '640px';
        camera.style.height = '480px';
        scene.appendChild(camera);

        let enableCam = document.createElement('button')
        enableCam.setAttribute('id', 'enableCam');
        enableCam.innerHTML = 'Enable Webcam';
        // enableCam.addEventListener("click", _enableCam);
        enableCam.addEventListener("click", function() {
          console.log("---click2---");
        })
        // console.log("add lister for enableCam");
        scene.appendChild(enableCam);


        for (let i = 0; i < self.classButtons.length; i ++) {
            scene.appendChild(self.classButtons[i]);
        }
        
        let trainButton = document.createElement('button');
        trainButton.setAttribute('id', 'train');
        trainButton.innerHTML = 'Train / Predict';
        trainButton.addEventListener('click', _trainAndPredict);
        scene.appendChild(trainButton);

        let resetButton = document.createElement('button');
        resetButton.setAttribute('id', 'reset');
        resetButton.addEventListener('click', _reset);
        resetButton.innerHTML = 'reset';
        scene.appendChild(resetButton);
        document.getElementById('quando_mlvision').append(scene);

    }

    return scene
  }


  function _gatherDataForClass() {

  }

  function _dataGatherLoop() {

  }

  function _trainAndPredict() {

  }

  function _reset() {
    
  }







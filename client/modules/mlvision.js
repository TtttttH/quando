import * as destructor from "./destructor.js";

const quando = window['quando']
const document = window['document']
const MOBILE_NET_INPUT_WIDTH = 224;
const MOBILE_NET_INPUT_HEIGHT = 224;
const STOP_DATA_GATHER = -1;
const STATUS = document.getElementById('quando_ml_status');
const VIDEO = document.getElementById('ml_webcam');
const ENABLE_CAM_BUTTON = document.getElementById('ml_enableCam');
const TRAIN_BUTTON = document.getElementById('ml_train');
const RESET_BUTTON = document.getElementById('reset');



  if (!quando) {
    alert('Fatal Error: ML_vision must be included after quando_browser')
  }

  let self = quando.mlvision = {}
  let mobilenetModel = undefined;


  self.trainDataInputs = [];    // set of training input data for machine learning model
  self.trainDataOutputs = [];
  self.examplesCount = [];
  self.classNames = [];
  self.classButtons = [];
  self.predict = false;
  self.mobilenet = undefined;
  self.classCounts = 0;
  self.gatherDataState = STOP_DATA_GATHER;
  self.videoPlaying = false;
  self.model = undefined;

  self.width = window.innerWidth
  self.height = window.innerHeight
  
  self.buildModel = () => {
    ENABLE_CAM_BUTTON.addEventListener('click', _enableCam);
    // set status
    STATUS.innerHTML = 'awaiting model lode';
    TRAIN_BUTTON.addEventListener('click', _trainAndPredict);
    RESET_BUTTON.addEventListener('click', _reset);
    _createClassButtons();
    _loadMobileNetFeatureModel();
    _loadTransferModel();
  }

  self.addClass = (class_name) => {
    self.classNames.push(class_name);
    let class_button = document.createElement('button');
    class_button.className = 'quando_ml_button';
    // class_button.className = 'quando_label';

    class_button.setAttribute("data-1hot", self.classCounts);
    class_button.setAttribute("data-name", class_name);
    class_button.innerHTML = "Gather " + class_name + " image data";
    class_button.addEventListener('mousedown', _gatherDataForClass);
    class_button.addEventListener('mouseup', _gatherDataForClass);
    self.classButtons.push(class_button);
    self.classCounts += 1;
  }

  function _createClassButtons() {
    let class_list = document.getElementById('mlvision_button_list');
    let div = document.getElementById('quando_mlvision');
    if (class_list == null) {
        class_list = document.createElement('div');
        class_list.setAttribute('id', 'mlvision_button_list');


        for (let i = 0; i < self.classButtons.length; i ++) {
            class_list.appendChild(self.classButtons[i]);
            let br = document.createElement('br');
            class_list.appendChild(br);
        }
        
        div.append(class_list);
    }

    return class_list
  }

  function _enableCam() {
    // alert("added");
    if (_hasGetUserMedia()) {
        // getUsermedia parameters.
        console.log("get video and button");
        const constraints = {
            video: true,
            width: 640,
            height: 480
        };

        // Activate the webcam stream.
        navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
            VIDEO.srcObject = stream;
            VIDEO.addEventListener('loadeddata', function() {
                self.videoPlaying = true;
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

  function _gatherDataForClass() {
    let classNumber = parseInt(this.getAttribute('data-1hot'));
    self.gatherDataState = (self.gatherDataState == STOP_DATA_GATHER) ? classNumber : STOP_DATA_GATHER;
    _dataGatherLoop();
  }

  function _dataGatherLoop() {
    if (self.videoPlaying && self.gatherDataState != STOP_DATA_GATHER) {
      let imageFeatures = tf.tidy(function() {
        let videoFrameAsTensor = tf.browser.fromPixels(VIDEO);
        let resizedTensorFrame = tf.image.resizeBilinear(videoFrameAsTensor, [MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH], true);
        let normalizedTensorFrame = resizedTensorFrame.div(255);
        return mobilenetModel.predict(normalizedTensorFrame.expandDims()).squeeze();
      });

      self.trainDataInputs.push(imageFeatures);
      self.trainDataOutputs.push(self.gatherDataState);

      if (self.examplesCount[self.gatherDataState] === undefined) {
        self.examplesCount[self.gatherDataState] = 0;
      }

      self.examplesCount[self.gatherDataState] ++;
      STATUS.innerText = '';

      for (let n = 0; n < self.classCounts; n ++) {
        STATUS.innerText += self.classNames[n] + 'data count:' + self.examplesCount[n] + '. ';
      }

      window.requestAnimationFrame(_dataGatherLoop);
    }
  }

  async function _trainAndPredict() {
    self.predict = false;
    tf.util.shuffleCombo(self.trainDataInputs, self.trainDataOutputs);
    let outputsAsTensor = tf.tensor1d(self.trainDataOutputs, 'int32');
    let oneHotOutputs = tf.oneHot(outputsAsTensor, self.classCounts);
    let inputsAsTensor = tf.stack(self.trainDataInputs);

    let results = await self.model.fit(inputsAsTensor, oneHotOutputs, {shuffle: true, batchSize: 5, epoches: 10, callbacks:{onEpochEnd: _logProgress} });

    outputsAsTensor.dispose();
    oneHotOutputs.dispose();
    inputsAsTensor.dispose();
    self.predict = true;
    _predictLoop();

  }

  function _predictLoop() {
    if (self.predict) {
      tf.tidy(function() {
        let videoFrameAsTensor = tf.browser.fromPixels(VIDEO).div(255);
        let resizedTensorFrame = tf.image.resizeBilinear(videoFrameAsTensor, [MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH], true);

        let imageFeatures = mobilenetModel.predict(resizedTensorFrame.expandDims());
        let prediction = self.model.predict(imageFeatures).squeeze();
        let highestIndex = prediction.argMax().arraySync();
        let predictionArray = prediction.arraySync();

        STATUS.innerText = 'Prediction: ' + self.classNames[highestIndex] + 'with' + Math.floor(predictionArray[highestIndex] * 100) + '% confidence';
      });

      window.requestAnimationFrame(_predictLoop);
    }
  }

  function _logProgress(epoch, logs) {
    console.log('Data for epoch' + epoch, logs);
  }


  async function _loadMobileNetFeatureModel() {
    const URL = 'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1';
    mobilenetModel = await tf.loadGraphModel(URL, {fromTFHub: true});
    STATUS.innerText = 'mobileNet loaded successfully';

    tf.tidy(function () {
      let answer = mobilenetModel.predict(tf.zeros([1, MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH, 3]));
      console.log(answer.shape);
    });
  }

  function _loadTransferModel() {
    self.model = tf.sequential();
    self.model.add(tf.layers.dense({inputShape: [1024], units: 128, activation: 'relu'}));
    self.model.add(tf.layers.dense({units: self.classCounts, activation: 'softmax'}));
    self.model.summary();

    self.model.compile({
      optimizer: 'adam',

      loss: (self.classCounts == 2) ? 'binaryCrossentropy' : 'categoricalCrossentropy',

      metrics: ['accuracy']
    });

    console.log('transfer model loaded');
  }


  function _reset() {
    self.predict = false;
    self.examplesCount.length = 0;
    for (let i = 0; i < self.trainDataInputs.length; i ++) {
      self.trainDataInputs[i].dispose();
    }

    self.trainDataInputs.length = 0;
    self.trainDataOutputs.length = 0;
    STATUS.innerText = 'No data collected';

    console.log('Tensors in memory:' + tf.memory().numTensors);
  }
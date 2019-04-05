(function () {
    var quando = this['quando']
    if (!quando) {
      alert('Fatal Error: AR must be included after quando_browser')
    }
    var self = quando.ar = {}
    self.width = screen.width
    self.height = screen.height

    self.initScene = function() {
      //init scene
      scene = document.createElement('a-scene');
      scene.setAttribute('arjs', 'debugUIEnabled: false;');
      scene.setAttribute('embedded', '');
      scene.setAttribute('id', 'scene');
      return scene;
    }

    self.initHiddenCanvas = function() {
      //init hidden canvas - used for drawing snapshot of webcam feed
      let hiddenCanvas = document.createElement('canvas');
      hiddenCanvas.setAttribute('id', 'hiddenCanvas');
      hiddenCanvas.setAttribute('width', 250);
      hiddenCanvas.setAttribute('height', 200);
      return hiddenCanvas;
    }

    self.initMarker = function(markerID) {
      let marker = document.getElementById(markerID);

      if (marker!=null) {
        alert('Marker '+markerID+' already exists!');
      } else {
        //init marker

        marker = document.createElement('a-marker');
        if (markerID == 'hiro') {
          marker.setAttribute('preset', 'hiro');
          marker.setAttribute('id', markerID);
        } else { 
          marker.setAttribute('preset', 'custom');
          marker.setAttribute('type', 'pattern');
          marker.setAttribute('id', markerID);
          //NOTE: below URLs must be hosted online instead of relatively for some dumb reason
          marker.setAttribute('url', 'https://raw.githubusercontent.com/andrewfstratton/quando/ar_dev/client/media/markers/'+markerID+'.patt');
        }
      }
      return marker;
    }

    self.initCam = function() {
      //init camera element
      var cam = document.createElement('a-camera-static'); 
      cam.setAttribute('id', 'camera');
      return cam;
    }

    self.whenMarker = function(markerID, orientation, fn) {
      let scene = document.getElementById('scene');
      if (scene == null) { //if scene DOES NOT exist
        let scene = self.initScene();
        let hiddenCanvas = self.initHiddenCanvas();
        let marker = self.initMarker(markerID);
        let camera = self.initCam();

        //add all elements to DOM
        scene.appendChild(marker);
        scene.appendChild(camera);
        document.getElementById('quando_AR').append(scene);
        document.body.appendChild(hiddenCanvas);
        
        //add onScan eventListener
        marker.addEventListener('markerFound', (e)=>{
          fn();
        });
      } else { //scene DOES exist
        let marker = self.initMarker(markerID);
        //add all elements to DOM
        scene.appendChild(marker);
        document.getElementById('quando_AR').append(scene);

        //add onScan eventListener
        marker.addEventListener('markerFound', (e)=>{
          fn();
        });
      }

    }

    self.showGLTF = function(markerID, modelURL, scale = 100, above) { //scale set to 10 for default
      //handle params
      scale = scale/100      
      if (above) { //is model to be on or above marker?
        position = '0 1 0'
      } else {
        position = '0 0 0'
      }
      if (modelURL == null) {
        alert('No model selected!');
      };
      modelURL = '/client/media/' + encodeURI(modelURL);

      let scene = document.getElementById('scene');

      if (scene == null) { //if scene DOES NOT exist
        alert ('You need to be in AR to show this model!');
      } else {
        let model = document.getElementById(modelURL+markerID);
        let marker = document.getElementById(markerID);
        if (model != null) { //if model DOES already exist
        } else {
          //init user chosen model - GLTF 2.0 - uncompressed
          model = document.createElement('a-gltf-model');
          model.setAttribute('gltf-model', 'url('+modelURL+')'); //id model from url
          model.setAttribute('scale', scale.toString() + ' '+ scale.toString() +' '+ scale.toString());
          model.setAttribute('position', position);
          model.setAttribute('id', (modelURL+markerID));
          //add to heirarchy
          marker.appendChild(model);
        }
      }
    }

    self.showGLTFOLD = function(modelURL, markerID, flat=true, scale=100, above=false) {
      //handle params
      if (modelURL == null) {
        alert('No model selected!')
      }
      modelURL = '/client/media/' + encodeURI(modelURL)
      
      scale = scale/100; //scale supplied in %
      if (above) { //is model to be on or above marker?
        position = '0 1 0'
      } else {
        position = '0 0 0'
      }

      var scene = document.getElementById('scene')
      if (scene == null) { 
        //if scene doesn't exist
        //init scene
        scene = document.createElement('a-scene');
        scene.setAttribute('arjs', 'debugUIEnabled: false;');
        scene.setAttribute('embedded', '');
        scene.setAttribute('id', 'scene');

        let hiddenCanvas = document.createElement('canvas');
        hiddenCanvas.setAttribute('id', 'hiddenCanvas');
        hiddenCanvas.setAttribute('width', self.width);
        hiddenCanvas.setAttribute('height', self.height);

        //init marker
        var marker = document.createElement('a-marker');
        if (markerID == 'hiro') {
          marker.setAttribute('preset', 'hiro');
          marker.setAttribute('id', 'hiro');
        } else { 
          marker.setAttribute('preset', 'custom');
          marker.setAttribute('type', 'pattern');
          marker.setAttribute('id', markerID);
          //NOTE: below URLs must be hosted online instead of relatively for some dumb reason
          marker.setAttribute('url', 'https://raw.githubusercontent.com/andrewfstratton/quando/ar_dev/client/media/markers/'+markerID+'.patt');
        }

        //init user chosen model - GLTF 2.0 - uncompressed
        var model = document.createElement('a-gltf-model');
        model.setAttribute('gltf-model', 'url('+modelURL+')'); //id model from url
        model.setAttribute('scale', scale.toString() + ' '+ scale.toString() +' '+ scale.toString());
        model.setAttribute('position', position);
        if (flat == false) {
          model.setAttribute('rotation', '0 0 90')
        }

        //init camera element
        var cam = document.createElement('a-camera-static'); 
        cam.setAttribute('id', 'camera')
        
        //add to heirarchy
        marker.appendChild(model);
        scene.appendChild(marker);
        scene.appendChild(cam);
        scene.appendChild(hiddenCanvas);

        //add to AR element of doc
        document.getElementById('quando_AR').append(scene);

      } else { //scene already exists

        //init marker
        var marker = document.createElement('a-marker');
        if (markerID == 'hiro') {
          marker.setAttribute('preset', 'hiro');
          marker.setAttribute('id', 'hiro');
        } else { 
          marker.setAttribute('preset', 'custom');
          marker.setAttribute('type', 'pattern');
          marker.setAttribute('id', markerID);
          //NOTE: below URLs must be hosted online instead of relatively for some dumb reason
          marker.setAttribute('url', 'https://raw.githubusercontent.com/andrewfstratton/quando/ar_dev/client/media/markers/'+markerID+'.patt');
        }

        //user chosen model - GLTF 2.0 - uncompressed
        var model = document.createElement('a-gltf-model');
        model.setAttribute('gltf-model', 'url('+modelURL+')'); //id model from url
        model.setAttribute('scale', scale.toString() + ' '+ scale.toString() +' '+ scale.toString());
        model.setAttribute('position', position);
        if (flat == false) {
          model.setAttribute('rotation', '0 0 90')
        }

        marker.appendChild(model)
        scene.appendChild(marker)

      }
    }

    self.showImage = function(markerID, imgURL, scale = 100, orientation) { //scale set to 0.1 for default
      scale = scale/100; //scale supplied in %

      //handle params
      if (imgURL == null) {
        alert('No image selected!');
      };
      imgURL = '/client/media/' + encodeURI(imgURL);

      let scene = document.getElementById('scene');

      if (scene == null) { //if scene DOES NOT exist
        alert ('You need to be in AR to show this image!');
      } else {
        let img = document.getElementById(imgURL+markerID);
        let marker = document.getElementById(markerID);
        if (img != null) { //if model DOES already exist
          alert('Image '+imgURL+' is already shown!');  
        } else {
          //init user chosen image
          img = document.createElement('a-image');
          img.setAttribute('src', imgURL); //id model from url
  
          //the below width and height settings are not relative, so will display the image in a 1:1 ratio
          img.setAttribute('height', scale.toString());
          img.setAttribute('width', scale.toString());
          if (orientation == 'flat') {
            img.setAttribute('rotation', '-90 0 0');
          } else if (orientation == 'vertical') {
            img.setAttribute('rotation', '0 0 0');
          } else {
            img.setAttribute('rotation', '0 0 0');
            img.setAttribute('look-at', '#player')
          }
        }
        marker.appendChild(img);
      }
    }

    self.showVideo = function(markerID, vidURL, scale = 100, orientation) { //scale set to 0.1 for default
      scale = scale/100; //scale supplied in %

      //handle params
      if (vidURL == null) {
        alert('No image selected!');
      };
      vidURL = '/client/media/' + encodeURI(vidURL);

      let scene = document.getElementById('scene');

      if (scene == null) { //if scene DOES NOT exist
        alert ('You need to be in AR to show this image!');
      } else {
        let vid = document.getElementById(vidURL+markerID);
        let marker = document.getElementById(markerID);
        if (vid != null) { //if model DOES already exist
          alert('Image '+vidURL+' is already shown!');  
        } else {
          //init user chosen image
          vid = document.createElement('a-video');
          vid.setAttribute('src', vidURL); //id model from url
  
          //the below width and height settings are not relative, so will display the image in a 1:1 ratio
          vid.setAttribute('height', scale.toString());
          vid.setAttribute('width', scale.toString());
          if (orientation == 'flat') {
            vid.setAttribute('rotation', '-90 0 90');
          } else if (orientation == 'vertical') {
            vid.setAttribute('rotation', '0 0 0');
          } else {
            vid.setAttribute('rotation', '0 0 0');
            vid.setAttribute('look-at', '#player')
          }
        }
        marker.appendChild(vid);
      }
    }


    self.showText = function(markerID, text, scale) {
      scale = scale/10; //scale supplied in %

      //handle params

      let scene = document.getElementById('scene');

      if (scene == null) { //if scene DOES NOT exist
        alert ('You need to be in AR to show this image!');
      } else {
        let textElem = document.getElementById(text+markerID);
        let marker = document.getElementById(markerID);
        if (textElem != null) {
        } else {
          //init user text
          textElem = document.createElement('a-text');
          textElem.setAttribute('value', text);
  
          //the below width and height settings are bad
          textElem.setAttribute('height', scale.toString());
          textElem.setAttribute('width', scale.toString());
  
        marker.appendChild(textElem);
        }
      }
    }

    self.clear = function() {
      var arDiv = document.getElementById('quando_AR')

      /*Get all video elements, then delete the one that's
      direct parent is the body.*/
      var videos = document.getElementsByTagName('video')
      for (i=0; i<videos.length; i++){
        if (videos[i].parentNode == body) {
          body.removeChild(videos[i])
        }
      }

      //clear AR div
      while (arDiv.firstChild) {
        arDiv.removeChild(arDiv.firstChild);
      }

      //TODO -- delete VR button w attribute 'AR-injected'
    }

    //adds event listener to marker that fires when the marker's scanned, executing the 
    //blocks in the box
    self.onLoss = function(markerID, delay, fn) {
      var marker = document.getElementById(markerID)
			marker.addEventListener('markerLost', (e)=>{
        fn()
      });
    }

}) ()
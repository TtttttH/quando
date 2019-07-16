(function () {
  var self = this['quando'] = {}
  self.idle_reset_ms = 0
  self.idle_callback_id = 0
  self._displays = new Map()
  self.pinching = false

  var _lookup = {} // holds run time arrays

  self.socket = io.connect('http://' + window.location.hostname)

  var websockets = []
  
  // self.call_tts = function(text, val) {        
  //   if (typeof val === 'string' && val.length) {
  //     text = val
  //   }
  //   //send POST request to server
  //   fetch('/watson/TTS_request', { method: 'POST', 
  //       body: JSON.stringify({'text':text}), 
  //       headers: {"Content-Type": "application/json",
  //                 Accept: 'application/json'}
  //   }).then(function(response) {
  //     response.json().then(function(data) {
  //       console.log(data)
  //       //play audio after 
  //       window.setTimeout(()=> {
  //         self.audio(data+'.wav', false)
  //       }, 0)
  //     })
  //   })
  // } 
    
  // self.call_ass = function(text) {
  //   //send POST request to server
  //   fetch('/watson/ASS_request', { method: 'POST', 
  //       body: JSON.stringify({'text':text}), 
  //       headers: {"Content-Type": "application/json",
  //                 Accept: 'application/json'}
  //   }).then(function(response) {
  //     console.log(response)
  //     let output = JSON.parse(response)
  //     self.text(output.generic[0].text, true)
  //   })
  // } 

  // self.call_vis_rec = function(goalClass, fn) {
  //   let scene = document.getElementById('scene')
  //   if (scene == null) { 
  //     //if scene DOES NOT exist
  //     let scene = self.ar.initScene()
  //     let hiddenCanvas = self.ar.initHiddenCanvas()
  //     let camera = self.ar.initCam()

  //     //add all elements to DOM
  //     scene.appendChild(camera)
  //     document.getElementById('quando_AR').append(scene)
  //     document.getElementById('quando_AR').appendChild(hiddenCanvas)
  //   }

  //   //now the scene deffo does exist
  //   const canvas = document.getElementById('hiddenCanvas')
  //   const context = canvas.getContext('2d')
  //   //placeholder variables
  //   let vid = null
  //   let imgData = null

  //   //timeout of 3 seconds to wait for webcam feed to load in
  //   window.setTimeout(()=>{
  //     let div = document.getElementById('visrec_label')
  //     let elem = document.getElementById('quando_labels')

  //     //if label doesn't already exist, create label
  //     if (div == null) {
  //       div = document.createElement('div')
  //       div.className = 'quando_label'
  //       div.innerHTML = "Found the thing?"
  //       div.setAttribute('id', 'visrec_label')
  //     }

  //     /*Get all video elements, then find the one that's
  //     direct parent is the body, which must be the webcam
  //     feed generated by AR.js*/
  //     var videos = document.getElementsByTagName('video')
  //     for (i=0; i<videos.length; i++){
  //       if (videos[i].parentNode == body) {
  //         vid = videos[i]
  //       }
  //     }
  //     if (vid == null) {
  //       alert('Webcam feed not available!')
  //     }

  //     div.addEventListener("click", function(){
  //       //save snapshot of webcam feed as dataURL
  //       context.drawImage(vid, 0, 0, 250, 200)
  //       imgData = canvas.toDataURL("image/png")
  //       //call api using this snapshot as input
  //       div.innerHTML = "Working..."
  //       self.call_vis_rec_api(imgData, goalClass, fn)
  //     })
  //     elem.appendChild(div)
  //   }, 3000)
  // }

  // self.call_vis_rec_api = function(imgData, goalClass, fn) {
  //       //send POST request to server
  //       fetch('/watson/VISREC_request', { method: 'POST', 
  //       body: JSON.stringify({'imgData':imgData}), 
  //       headers: {"Content-Type": "application/json"}
  //     }).then(
  
  //       //once POST request is done
  //       function(response) {

  //         let div = document.getElementById('visrec_label')
  //         div.innerHTML = "Seen! Click to try again if you want to..."

  //         response.json().then(function(data) {
  //           //has it seen what it wants to?
  //           if (data.includes(goalClass)) {
  //             //execute box
  //             fn()
  //           }
  //         })
  //       }
  //     )
  // }
  
  // self.call_speech_to_text = function() {  
  //   let div = document.getElementById('visrec_label')
  //   let elem = document.getElementById('quando_labels')
  //   let recording = false
  //   let mediaRecorder = null
  //   //if label doesn't already exist, create label
  //   if (div == null) {
  //     div = document.createElement('div')
  //     div.className = 'quando_label'
  //     div.innerHTML = "Click to start listening..."
  //     div.setAttribute('id', 'stt_label')
  //   }

  //   navigator.mediaDevices.getUserMedia({ audio: true, video: false })
  //   .then(stream => {
  //     mediaRecorder = new MediaRecorder(stream)
  //     const audioChunks = []
  
  //     mediaRecorder.ondataavailable = e => {
  //       audioChunks.push(e.data)
  //       if (mediaRecorder.state == "inactive") {
  //         const audioBlob = new Blob(audioChunks,{type:'video/webm'});
  //         var reader = new window.FileReader();
  //         reader.readAsDataURL(audioBlob); 
  //         reader.onloadend = function() {
  //            base64 = reader.result;
  //            base64 = base64.split(',')[1];
  //            console.log(base64 );
  //            fetch('/watson/SPEECH_request', { method: 'POST', 
  //              body: JSON.stringify({'data':base64}), 
  //              headers: {"Content-Type": "application/json"}
  //            }).then(function(response) {
  //               response.json().then(function(data) {
  //                 console.log(data.replace(/"/g, ""))
  //                 let input = document.getElementById('inp')
  //                 input.value = data.replace(/"/g, "")
  //                 input.click()
  //                 div.innerHTML = "Click to start listening..."
  //               })
  //             })
  //         }
  //       }
  //     }

  //   })

  //   self.call_speech_to_text_ass = function(fn) {  
  //     let div = document.getElementById('visrec_label')
  //     let elem = document.getElementById('quando_labels')
  //     let recording = false
  //     let mediaRecorder = null
  //     //if label doesn't already exist, create label
  //     if (div == null) {
  //       div = document.createElement('div')
  //       div.className = 'quando_label'
  //       div.innerHTML = "Click to start listening..."
  //       div.setAttribute('id', 'stt_label')
  //     }
  
  //     navigator.mediaDevices.getUserMedia({ audio: true, video: false })
  //     .then(stream => {
  //       mediaRecorder = new MediaRecorder(stream)
  //       const audioChunks = []
    
  //       mediaRecorder.ondataavailable = e => {
  //         audioChunks.push(e.data)
  //         if (mediaRecorder.state == "inactive") {
  //           const audioBlob = new Blob(audioChunks,{type:'video/webm'});
  //           var reader = new window.FileReader();
  //           reader.readAsDataURL(audioBlob); 
  //           reader.onloadend = function() {
  //              base64 = reader.result;
  //              base64 = base64.split(',')[1];
  //              console.log(base64 );
  //              fetch('/watson/SPEECH_request', { method: 'POST', 
  //                body: JSON.stringify({'data':base64}), 
  //                headers: {"Content-Type": "application/json"}
  //              }).then(function(response) {
  //                 response.json().then(function(data) {
  //                   console.log(data.replace(/"/g, ""))
  //                   //let input = document.getElementById('inp')
  //                   //input.value = data.replace(/"/g, "")
  //                   //input.click()
  //                   fn(data.replace(/"/g, ""))
  //                   div.innerHTML = "Click to start listening..."
  //                 })
  //               })
  //           }
  //         }
  //       }
  
  //     })
  //   }

  //   div.addEventListener("click", function(){
  //     if (!recording) {
  //       mediaRecorder.start()
  //       div.innerHTML = "Stop listening..."
  //       recording = true
  //     } else {
  //       recording = false
  //       mediaRecorder.stop()
  //       div.innerHTML = "Working..."
  //     }
  //   }) 


  //   elem.appendChild(div)
  //   /*/send POST request to server*/
  // }


  // self.addToneHandler = function(goalTone, fn) {
  //   //adds 'onClick' event listener to the input prompt button
  //   //which submits the prompts text to the tone analyzer
  //   let input = document.getElementById('inp')
  //   let button = document.getElementById('inpButton')

  //   button.addEventListener("click", function(){
  //     //send POST request to server
  //     fetch('/watson/TONE_request', { method: 'POST', 
  //       body: JSON.stringify({'text':input.value}), 
  //       headers: {"Content-Type": "application/json"}
  //     }).then(function(response) {
  //       response.json().then(function(data) {
  //         //if tone of input is goal
  //         if (data.includes(goalTone)) {
  //           //execute box
  //           fn()
  //         }
  //       })
  //     })
  //   })
  // }

  function _displayWidth() {
    return window.innerWidth
  }

  function _displayHeight() {
    return window.innerHeight
  }

  self.socket.on('deploy', function (data) {
    var locStr = decodeURIComponent(window.location.href)
    if (locStr.endsWith('/'+data.script)) {
      window.location.reload(true) // nocache reload - probably not necessary
    }
  })

  self.add_message_handler = function (message, fn) {
    self.socket.on(message, fn)
    self.destructor.add( () => {
      self.socket.off(message, fn)
    })
  }

  self.send_message = function(message, val, host='') {
    fetch('/message/' + message, { method: 'POST', 
      body: JSON.stringify({'val':val, 'host':host}), 
      headers: {"Content-Type": "application/json"}
    })
  }

  self.create_websocket_connection = function (url) {
    var socket = self.get_websocket_connection(url)

    if (!socket) {
      socket = new WebSocket(url)
      socket.onerror = err => console.error("Websocket connection error: ", err)

      websockets.push(socket)
    }

    return socket
  }

  self.get_websocket_connection = function (url) {
    for (i = 0; i < websockets.length; i++) {
      if (websockets[i].url == url) { return websockets[i] }
    }
  }

  self.send_websocket_message = function (output, url, val) {
    if (val) { output = val }

    var socket = self.create_websocket_connection(url)

    if (socket.readyState == WebSocket.OPEN) {
      socket.send(output)
    } else {
      setTimeout(() => self.send_websocket_message(output, url, val), 100)
    }
  }

  self.add_websocket_handler = function (url, callback) {
    var socket = self.create_websocket_connection(url)

    socket.onmessage = (evt) => {
      if (typeof callback === 'function') { callback(evt.data) }
    }

    self.destructor.add( () => {
      if (socket.readyState == WebSocket.OPEN) {
        socket.close()
      }

      websockets.splice(websockets.indexOf(socket), 1)
    })
  }

  self.idle_reset = function () {
    if (self.idle_reset_ms > 0) {
      clearTimeout(self.idle_callback_id)
      self.idle_callback_id = setTimeout(self.idle_callback, self.idle_reset_ms)
    } else { // this means we are now idle and must wakeup
      if (self.idle_active_callback) {
        self.idle_active_callback()
      }
    }
  }

  document.onmousemove = self.idle_reset
  document.onclick = self.idle_reset
  document.onkeydown = self.idle_reset
  document.onkeyup = self.idle_reset

  self.dispatch_event = function (event_name, data = false) {
    if (data) {
      document.dispatchEvent(new CustomEvent(event_name, data))
    } else {
      document.dispatchEvent(new CustomEvent(event_name))
    }
  }

  self.add_handler = function (event, callback) {
    document.addEventListener(event, callback)
    self.destructor.add( () => {
      document.removeEventListener(event, callback)
    })
  }

  self.add_scaled_handler = function (event_name, callback, scaler) {
    let last_value = null
    var handler = function (ev) {
      var value = scaler(ev.detail)
      if (value !== null) {
        if (value != last_value) {
          last_value = value
          callback(value)
        }
      }
    }
    quando.add_handler(event_name, handler)
  }

  self.new_scaler = function (min, max, inverted = false) {
    return function (value) {
      // convert to range 0 to 1 for min to max
      var result = (value - min) / (max - min)
      result = Math.min(1, result)
      result = Math.max(0, result)
      if (inverted) {
        result = 1 - result
      }
      return result
    }
  }

  self.new_angle_scaler = function (mid, plus_minus, inverted = false) {
    var mod = function(x, n) {
        return ((x%n)+n)%n
    }
    var last_result = 0
    var crossover = mod(mid+180, 360)
    // i.e. 25% of the non used range
    var crossover_range = (180 - Math.abs(plus_minus)) / 4
    return function (value) {
      var x = mod(value - mid, 360)
      if (x > 180) { x -= 360}
      var result = (x + plus_minus) / (2 * plus_minus)
      if (inverted) {
        result = 1 - result
      }
      if ((result < 0) || (result > 1)) { // i.e. result is out of range
            // identify if result should be used
            var diff = Math.abs(crossover - mod(value, 360))
            if (diff <= crossover_range) { // inside crossover range, so use the last result 
                result = last_result
            }
      }
      result = Math.min(1, result)
      result = Math.max(0, result)
      last_result = result
      return result
    }
  }

  // Start in the middle
  self._y = _displayHeight()
  self._x = _displayWidth()

  function _cursor_adjust () {
    var x = self._x
    var y = self._y
    var style = document.getElementById('cursor').style
    var max_width = _displayWidth()
    var max_height = _displayHeight()
    if (x < 0) {
      x = 0
    } else if (x > max_width) {
      x = max_width
    }
    if (y < 0) {
      y = 0
    } else if (y > max_height) {
      y = max_height
    }
    style.top = y + 'px'
    style.left = x + 'px'
    style.visibility = 'hidden' // TODO this should only be done once - maybe an event (so the second one can be consumed/ignored?)
    var elem = document.elementFromPoint(x, y)
    style.visibility = 'visible'
    self.hover(elem)
    self.idle_reset()
  }

  self.cursor_up_down = function (mid, range, inverted, y) {
    min = mid-range
    max = mid+range
    min /= 100
    max /= 100
    if (y === false) {
      y = (min + max)/2
    }
    if (!inverted) {
      y = 1 - y // starts inverted
    }
    var scr_min = min * _displayHeight()
    var scr_max = max * _displayHeight()
    self._y = scr_min + (y * (scr_max-scr_min))
    _cursor_adjust()
  }

  self.cursor_left_right = function (mid, range, inverted, x) {
    min = mid-range
    max = mid+range
    min /= 100
    max /= 100
    if (x === false) {
      x = (min + max)/2
    }
    if (inverted) {
      x = 1 - x // starts normal
    }
    var scr_min = min * _displayWidth()
    var scr_max = max * _displayWidth()
    self._x = scr_min + (x * (scr_max-scr_min))
    _cursor_adjust()
  }

  var Config = self.Config = {
  }

  self.idle = function (count, units, idle_fn, active_fn) {
    clearTimeout(self.idle_callback_id)
    let time_secs = self.idle_reset_ms = self.time.units_to_ms(units, count)
    self.idle_callback = () => {
      self.idle_reset_ms = 0 // why - surely I need to intercept self.idle_reset
            // actually - this will work to force self.idle_reset to call idle_active_callback instead
      idle_fn()
    }
    self.idle_active_callback = () => {
      clearTimeout(self.idle_callback_id)
      self.idle_reset_ms = time_secs // resets to idle detection
      self.idle_callback_id = setTimeout(self.idle_callback, self.idle_reset_ms)
            // so, restarts timeout when active
      active_fn()
    }
    self.idle_callback_id = setTimeout(self.idle_callback, self.idle_reset_ms)
  }

  function _set_or_append_tag_text(txt, tag, append) {
    var elem = document.getElementById(tag)
    if (typeof txt === 'function') {
      // HACK: N.B. This may be a security worry?!
      txt = txt()
    }    
    if (append) {
      txt = elem.innerHTML + txt
    }
    if (!txt) {
      elem.style.visibility = 'hidden'
      txt = ''
    } else {
      elem.style.visibility = 'visible'
    }
    elem.innerHTML = txt
  }
  
  self.title = function (text = '', append = false) {
    _set_or_append_tag_text(text, 'quando_title', append)
  }

  self.text = function (text = '', append = false, val) {
    if (typeof val === 'string' && val.length) {
      text = val
    }
    _set_or_append_tag_text(text, 'quando_text', append)
  }

  self.projection = function (front = true) {
    let scale = 1
    if (!front) {
      scale = -1
    }
    document.getElementById('html').style.transform = 'scale(' + scale + ',1)'
  }

  self.image_update_video = function (img) {
    var image = document.getElementById('quando_image')
    if (image.src != encodeURI(window.location.origin + img)) {
            // i.e. only stop the video when the image is different - still need to set the image style...
            // TODO this needs checking for behavioural side effects
      self.clear_video()
    }
  }

  self.video = function (vid, loop = false) {
    if (vid) {
      vid = '/client/media/' + encodeURI(vid)
    }
    var video = document.getElementById('quando_video')
    video.loop = loop
    if (video.src != encodeURI(window.location.origin + vid)) { // i.e. ignore when already playing
      if (video.src) {
        video.pause()
      }
      video.src = vid
      video.autoplay = true
      video.addEventListener('ended', self.clear_video)
      video.style.visibility = 'visible'
      video.load()
    }
  }

  self.clear_video = function () {
    var video = document.getElementById('quando_video')
    video.src = ''
    video.style.visibility = 'hidden'
    // Remove all event listeners...
    video.parentNode.replaceChild(video.cloneNode(true), video)
  }

  self.audio = function (audio_in, loop = false) {
    if (audio_in) {
      audio_in = '/client/media/' + encodeURI(audio_in)
    }
    var audio = document.getElementById('quando_audio')
    audio.loop = loop
    if (audio.src != encodeURI(window.location.origin + audio_in)) { // src include http://127.0.0.1/
      if (audio.src) {
        audio.pause()
      }
      audio.src = audio_in
      audio.autoplay = true
      audio.addEventListener('ended', self.clear_audio)
      audio.load()
    }
  }

  self.clear_audio = function () {
    var audio = document.getElementById('quando_audio')
    audio.src = ''
    // Remove all event listeners...
    audio.parentNode.replaceChild(audio.cloneNode(true), audio)
  }

  self.replay_video_audio = function () {
    let video = document.getElementById('quando_video')
    let audio = document.getElementById('quando_audio')
    let video_audio = null

    if (!video.paused) {
      video_audio = video
    } else if (!audio.paused) {
      video_audio = audio
    }

    if (video_audio) {
      video_audio.pause()
      video_audio.currentTime = 0
      video_audio.play()
    }
  }

  self.hands = function (count, do_fn) {
    var hands = 'None'
    var handler = function () {
      frame = self.leap.frame()
      if (frame.hands) {
        self.idle_reset() // any hand data means there is a visitor present...
        if (frame.hands.length !== hands) {
          hands = frame.hands.length
          if (hands === count) {
            do_fn()
          }
        }
      }
    }
    if (self.leap) {
      self.time.every({count: 1/20}, handler)
    } else {
      self.leap = new Leap.Controller()
      self.leap.connect()
      self.leap.on('connect', function () {
        self.time.every({count: 1/20}, handler)
      })
    }
  }

  self.handed = function (left, right, do_fn) {
    var handler = function () {
// FIX very inefficient...
      frame = self.leap.frame()
      var now_left = false
      var now_right = false
      if (frame.hands) {
        self.idle_reset() // any hand data means there is a visitor present...
        if (frame.hands.length !== 0) {
          var hands = frame.hands
          for (var i = 0; i < hands.length; i++) {
            var handed = hands[i].type
            if (handed === 'left') {
              now_left = true
            }
            if (handed === 'right') {
              now_right = true
            }
          }
        }
      }
      if ((now_right === right) && (now_left === left)) {
        do_fn()
      }
    }
    if (self.leap) {
      self.time.every({count: 1/20}, handler)
    } else {
      self.leap = new Leap.Controller()
      self.leap.connect()
      self.leap.on('connect', function () {
        self.time.every({count: 1/20}, handler)
      })
    }
  }

  self.display = function (key, fn) { // Yes this is all of it...
    self._displays.set(key, fn)
  }

  self._removeFocus = function () {
    var focused = document.getElementsByClassName('focus')
    for (var focus of focused) {
      focus.classList.remove('focus')
      focus.removeEventListener('transitionend', self._handle_transition)
    }
  }
  self._handle_transition = function (ev) {
    ev.target.click()
  }

  self.startDisplay = function () {
    setTimeout( () => {
      self.style.set(self.style.DEFAULT, '#cursor', 'background-color', 'rgba(255, 255, 102, 0.7)')
      self.style.set(self.style.DEFAULT, '#cursor', 'width', '4.4vw')
      self.style.set(self.style.DEFAULT, '#cursor', 'height', '4.4vw')
      self.style.set(self.style.DEFAULT, '#cursor', 'margin-left', '-2.2vw')    
      self.style.set(self.style.DEFAULT, '#cursor', 'margin-top', '-2.2vw')    
      document.querySelector('body').addEventListener('contextmenu', // right click title to go to setup
              function (ev) {
                ev.preventDefault()
                window.location.href = '../../client/setup'
                return false
              }, false)
      self.pinching = false
      exec() // this is the function added by the generator
      let first = self._displays.keys().next()
      if (first && !first.done) {
        self.showDisplay(first.value) // this runs the very first display :)
      }
    }, 0)
  }

  self.hover = function (elem) {
    if (elem) {
      if (!elem.classList.contains('focus')) { // the element is not in 'focus'
                // remove focus from all other elements - since the cursor isn't over them
        self._removeFocus()
        if (elem.classList.contains('quando_label')) {
          elem.classList.add('focus')
          elem.addEventListener('transitionend', self._handle_transition)
        }
      }
    } else {
            // remove focus from any elements - since the cursor isn't over any
      self._removeFocus()
    }
  }

  self.showDisplay = function (id) {
    // perform any destructors - which will cancel pending events, etc.
    // assumes that display is unique...
    self.destructor.destroy()
    // Clear current labels, title and text
    document.getElementById('quando_labels').innerHTML = ''
    self.title()
    self.text()
    //clear AR 
    quando.ar.clear()
//        self.video() removed to make sure video can continue playing between displays
    self.style.reset()
    // Find display and execute...
    self._displays.get(id)()
  }

  self.addLabel = function (id, title) {
    self.addLabelStatement(title, () => { setTimeout( () => { quando.showDisplay(id) }, 0) })
  }

  self.addLabelStatement = function (title, fn) {
    var elem = document.getElementById('quando_labels')
    var div = document.getElementById('label'+title)
    if (div == null) {
      div = document.createElement('div')
      div.className = 'quando_label'
      div.innerHTML = title
      div.setAttribute('id', 'label'+title)
      elem.appendChild(div)
      div.onclick = fn
    }
  }

  self.promptInput = function() {
    var elem = document.getElementById('quando_labels')
    var div = document.createElement('div')
    var input = document.createElement('input')
    var button = document.createElement('button')
    button.setAttribute('id', 'inpButton')
    input.setAttribute('id', 'inp')

    div.className = 'quando_label'
    input.type = "text"
    input.className = "quando_input"
    button.innerHTML = "Submit"

    div.appendChild(input)
    div.appendChild(button)
    elem.appendChild(div)
    
  }

  self.pick = function(val, arr) {
    if (val === false) {
      val = 0.5
    }
    var i = Math.floor(val * arr.length)
    if (i == arr.length) {
      i--
    }
    arr[i]()
  }

  self.pick_random = function(arr) {
    var r = Math.random()
    self.pick(r, arr)
  }

  self.pick_one_each_time = function(arr) {
    if (arr.length > 0) {
      if (!arr.hasOwnProperty('index')) {
        arr.index = 0
      }
      var fn = arr[arr.index]
      if (++arr.index >= arr.length) {
        arr.index = 0
      }
      if (typeof fn === 'function') { fn() }
    }
  }

  self.setOnId = (id, val) => {
    _lookup[id] = val
  }

  self.getOnId = (id) => {
    return _lookup[id]
  }

  self.vary_each_time = function(fn, end_value, inverted, seesaw) {
    end_value--
    if (end_value < 1) { // Avoid infinite answers...
      end_value = 1
    }
    if (!fn.hasOwnProperty('counter')) {
      fn.counter = 0
      fn.direction = 1 // go up at the start - inverted will flip the final result...
    }
    // calculate the current value, then adjust the counter to the next value...
    let val = fn.counter/end_value
    if (inverted) {
      val = 1 - val
    }
    fn.counter += fn.direction
    if (fn.counter < 0) {
      if (seesaw) {
        fn.counter = 1 // i.e. next time is one more than the 0 minimum
        fn.direction = 1
      } else {
        fn.counter = end_value // logically this can't happen now...
      }
    }
    if (fn.counter > end_value) {
      if (seesaw) {
        fn.counter = end_value - 1 // next value is one less than the maximum
        fn.direction = -1
      } else {
        fn.counter = 0 // back to start
      }
    }
    if (typeof fn === 'function') { fn.call(this, val) }
  }

  function _degrees_to_radians (degrees) {
      var radians = Math.PI * degrees / 180
      return radians
  }

  self.convert_angle = (val, mid, range, inverted) => {
    if (val === false) { val = 0.5 }
    if (inverted) { val = 1 - val }
    let min = _degrees_to_radians(mid - range)
    let max = _degrees_to_radians(mid + range)
    return min + (val * (max-min))
  }

  self.clear = (display, title, text, image, video, audio, object3d, speech) => {
    if (title) { self.title(display) }
    if (text) { self.text(display) }
    if (image) { self.image.set(display) }
    if (video) { self.video('',false) }
    if (audio) { self.audio('',false) }
    if (object3d) { self.object3d.clear() }
    if (speech) { self.speech.clear() }
  }
})()

var val = false // force handler to manage when not embedded

if (document.title == "[[TITLE]]") { // this was opened by Inventor >> Test
  document.title = "TEST"
  let script = document.getElementById("quando_script")
  if (script) {
    script.parentNode.removeChild(script)
  }
  let exec = window.opener.index.clientScript()
  eval("this['exec'] =  () => {\n" + exec + "\n}")
}

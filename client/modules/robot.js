import * as destructor from "./destructor.js";

const quando = window['quando']
if (!quando) {
    alert('Fatal Error: Robot must be included after quando_browser')
}
let self = quando.robot = {}
let state = { // holds the requested state
    hand : { left: {}, right: {}},
    head : { yaw: {}, pitch: {}},
    shoulder : { left: {roll: {}, pitch: {}}, right: {roll: {}, pitch: {}}},
    elbow : { left: {roll: {}, yaw: {}}, right: {roll: {}, yaw: {}}},
    wrist : { left: {yaw: {}}, right: {yaw: {}}},
    motion: {sequence:[], interrupt:true},
    mirror : false // when true, switches left/right, etc.
}
let session = false

const STEP_LENGTH = 25.0/1000 //in mm

class vocabList {
    constructor(listName, vocab) {
        this.listName = listName;
        this.vocab = vocab;
    }
}

let robot = {
    TextToSpeech: {
        CurrentWord: null,
        CurrentSentence: null,
        Status: 'done',
        TextDone: null
    },
    AudioPlayer: {
        playing: false
    }
}

self._list = []

function log_error(err) {
    console.log("Error: " + err)
    setTimeout(updateRobot, 1000) // Try again in a second...
}

function helper_ConvertAngleToRads(angle) { //from degree to rads
    return angle * (Math.PI / 180)
}

function set_up() {
    session.service("ALAutonomousLife").then(function (al) {
        al.setState('solitary').fail(log_error)

        session.service("ALMemory").then(function (ALMemory) {
            ALMemory.subscriber("AutonomousLife/State").then(function (sub) {
                sub.signal.connect(function (state) {
                    session.service("ALRobotPosture").then(function (rp) {
                        Promise.resolve(rp.getPosture()).then(conditional).then(function (value) {
                            if (!value && state == "disabled") {
                                rp.goToPosture("Stand", 1.0);
                            }
                        })
                    }).fail(log_error)
                });
            });
        });
    }).fail(log_error)
}

function nao_reconnect(robotIp) {
    setTimeout(() => { self.connect(robotIp) }, 1000)
}

function execute_event_listeners() {
    session.service("ALMemory").then(function (ALMemory) {
        ALMemory.subscriber("ALTextToSpeech/CurrentWord").then(function (sub) {
            sub.signal.connect(function (value) {
                robot.TextToSpeech.CurrentWord = value;
            });
        });
    });

    session.service("ALMemory").then(function (ALMemory) {
        ALMemory.subscriber("ALTextToSpeech/CurrentSentence").then(function (sub) {
            sub.signal.connect(function (value) {
                console.log(value);
                robot.TextToSpeech.CurrentSentence = value;
            });
        });
    });

    session.service("ALMemory").then(function (ALMemory) {
        ALMemory.subscriber("ALTextToSpeech/Status").then(function (sub) {
            sub.signal.connect(function (value) {
                robot.TextToSpeech.Status = value[1];
            });
        });
    });

    session.service("ALMemory").then(function (ALMemory) {
        ALMemory.subscriber("ALTextToSpeech/TextDone").then(function (sub) {
            sub.signal.connect(function (value) {
                robot.TextToSpeech.TextDone = value;
            });
        });
    });
}

function conditional(value) {
    if (value == "disabled" || value == "Stand")
        return true;
}

self.connect = (robotIP) => {
    session = new QiSession(robotIP)
    session.socket().on('connect', () => {
        console.log('QiSession connected!')
        set_up()
        execute_event_listeners()
    }).on('disconnect', () => {
        console.log('QiSession disconnected!')
        nao_reconnect(robotIP)
    }).on('connect_error', () => {
        nao_reconnect(robotIP)
    })
}

let audioSequence = []
let audioInterrupt = true

self.audio = (fileName, scope) => {
    let path = '/home/nao/audio/'

    if (scope == 'interrupt') audioSequence = []
    audioInterrupt = scope == 'interrupt'
    if (scope == 'background') {
        session.service('ALAudioPlayer').then(ap => {
            ap.playFile(path + fileName).fail(log_error)
        })
    } else {
        audioSequence.push(() => {
            session.service('ALAudioPlayer').then(ap => {
                robot.AudioPlayer.playing = true
                ap.playFile(path + fileName).fail(log_error).always(() => {
                    robot.AudioPlayer.playing = false
                })
            })
        })
    }
    
    destructor.add(function () {
        audioSequence = []
        audioInterrupt = true
    })
}

self.masterVolume = (volume) => {
    session.service('ALAudioDevice').then(ad => ad.setOutputVolume(volume)).fail(log_error)
}

self.speechHandlerOl = (anim, text, pitch, speed, doubleVoice, doubleVoiceLevel, doubleVoiceTimeShift, interrupt=false, val) => {
    if (typeof val === 'string' && val.length) {
        text = val
    }

    self.changeVoice(pitch, speed, doubleVoice, doubleVoiceLevel, doubleVoiceTimeShift)

    if (interrupt) audioSequence = []
    audioInterrupt = interrupt
    
    if (audioSequence.length > 15) {
    audioSequence = []
    }
    
    audioSequence.push(() => {
        if (anim == "None") {
            self.say(text)
        } else {
            self.animatedSay(anim, text)
        }
    })
    destructor.add(function () {
        audioSequence = []
        audioInterrupt = true
    })
}


self.speechHandler = (anim, text, pitch, speed, doubleVoice, doubleVoiceLevel, doubleVoiceTimeShift, interrupt, val) => {
    //overrie text if the block is being passe a value
    if (typeof val === 'string' && val.length) {
        text = val
    }

    // self.changeVoice(pitch, speed, doubleVoice, doubleVoiceLevel, doubleVoiceTimeShift)

    if (interrupt == 'full') { //if full interrupt erase audioSequence before saying
    audioSequence = []
    audioSequence.push(() => {
        if (anim == "None") {
        self.say(text)
        // console.log('robot: "' + text + "'")
        } else {
        self.animatedSay(anim, text)
        // console.log('robot: "' + text + "'")
        }
    })
    } else if (interrupt == 'continue') { //if continue put interruption at start of sequence
    audioSequence.unshift(() => {
        if (anim == "None") {
        self.say(text)
        // console.log('robot: "' + text + "'")
        } else {
        self.animatedSay(anim, text)
        // // console.log('robot: "' + text + "'")
        }
    })
    } else { //else just add to end of sequence
    console.log('none int' + text)
    audioSequence.push(() => {
        if (anim == "None") {
        self.say(text)
        // console.log('robot: "' + text + "'")
        } else {
        self.animatedSay(anim, text)
        // console.log('robot: "' + text + "'")
        }
    })
    }
    
    if (interrupt == 'full') {
    audioInterrupt = true
    }
    // if (audioSequence.length > 15) {
    //   audioSequence = []
    // }
    destructor.add(function () {
        audioSequence = []
        audioInterrupt = true
    })
}

self.changeVoice = (pitch, speed, doubleVoicePitch, doubleVoiceLevel, doubleVoiceTimeShift) => {
    session.service("ALTextToSpeech").then((tts) => {
        tts.setParameter("pitchShift", pitch)
        tts.setParameter("speed", speed)
        tts.setParameter("doubleVoice", doubleVoicePitch)
        tts.setParameter("doubleVoiceLevel", doubleVoiceLevel)
        tts.setParameter("doubleVoiceTimeShift", doubleVoiceTimeShift)
        // tts.setParameter("doubleVoice", 1)
        // tts.setParameter("doubleVoiceLevel", 0)
        // tts.setParameter("doubleVoiceTimeShift", 0.0)
        
    }).fail(log_error)
}

self.say = (text) => {
    session.service("ALTextToSpeech").then((tts) => {
        if (robot.TextToSpeech.CurrentSentence != text) {
            tts.say(text)
        }
    }).fail(log_error)
}

self.animatedSay = (anim, text) => {
    session.service("ALAnimatedSpeech").then((aas) => {
        if (anim == "Random") {
            aas.setBodyLanguageMode(1) //random body language
        } else {
            aas.setBodyLanguageMode(0) //contextual body language
        }

        if (robot.TextToSpeech.CurrentSentence != text) {
            aas.say(text)
        }
    }).fail(log_error)
}
    
self.playAnimationTag = (animTag) => {
    self.speechHandler('Contextual', '^startTag(' + animTag + ') ^waitTag(' + animTag + ')', 0, 0, 0, 0, 0, 'continue', val)
}

self.playSine = (frequency, gain, duration) => {
    session.service("ALAudioDevice").then((aadp) => {
        aadp.playSine(frequency, gain, 0, duration)
    }).fail(log_error)
} 

self.mirror = (mirror) => {
    state.mirror = mirror
}

const HEAD_ANGLES = {
    true:{direction:'yaw', min:2.0857, max:-2.0857},
    false:{direction:'pitch', min:0.5149, max:-0.6720}
}

self.turnHead = (yaw, min_percent, max_percent, speed, inverted, val) => {
    if (inverted) { val = 1-val }
    let {direction, min, max} = HEAD_ANGLES[yaw]
    if (state.mirror && yaw) {
        [max, min] = [min, max] // Warning: this likely only works for symmetric angles
    }
    let range = max - min
    // now range may be changed by the min and max percent
    max = (range*max_percent/100)+min // N.B. Do not put this statement after the next
    min += range*min_percent/100
    let radians = min + (val * (max-min))
    let new_state = state.head[direction]
    new_state.angle = radians
    new_state.speed = speed
}

const ARM_JOINT_ANGLES = { // Note: set for the right arm - looking out from the Nao
    'roll_shoulder':{joint:'shoulder', direction:'roll', min:0.3142, max:-1.3265, mirror:true},
    'pitch_shoulder':{joint:'shoulder', direction:'pitch', min:2.0857, max:-2.0857}, // max and min inverted so bigger is up
    'yaw_elbow':{joint:'elbow', direction:'yaw', min:-2.0857, max:2.0857, mirror:true},
    'roll_elbow':{joint:'elbow', direction:'roll', min:1.5446, max:0.0349, mirror:false, negate:true}, // only one to negate only...
    'yaw_wrist':{joint:'wrist', direction:'yaw', min:-1.8238, max:1.8238, mirror:true}
}

self.moveArmJoint = (left, joint_direction, min_percent, max_percent, speed, inverted, val) => {
    if (inverted) { val = 1-val }
    let {joint, direction, min, max, mirror=false, negate=false} = ARM_JOINT_ANGLES[joint_direction]
    if (state.mirror) {
    left = !left
    if (mirror) {
        val = 1-val // and value is swapped?!
    }
    if (negate) {
        [max, min] = [min, max] // Note: this logic makes sense eventually...
    }
    } // When mirrored, left and right are swapped?!
    if (left) {
        [max, min] = [-min, -max]
    }

    let range = max - min
    // now range may be changed by the min and max percent
    max = (range*max_percent/100)+min // N.B. Do not put this statement after the next
    min += range*min_percent/100
    let radians = min + (val * (max-min))
    let position = left?'left':'right'
    let new_state = state[joint][position][direction]
    new_state.angle = radians
    new_state.speed = speed
}


self.initVideoStream = () => {
    session.service("ALVideoDevice").then((avd) => {
        // console.log(avd.getActiveCamera())
        let cam = avd.subscribeCamera("subscriberID", 0, "kVGA", "kRGB", 10).then((camera) => {
            // console.log('TESst'+camera)
        }).fail(log_error) 
        //subscriberID, resolution, colour space, fps
        let results = avd.getImageRemote(cam)
        // console.log("res: "+results)
        imgData = results[6]
    }).fail(log_error)
}

function updateYawPitch(motion, joint, yaw_pitch) {
    if (yaw_pitch.hasOwnProperty('angle') && (yaw_pitch.angle !== yaw_pitch.last_angle)) { // update yaw
        motion.killTasksUsingResources([joint]) // Kill any current motions
        motion.setAngles(joint, yaw_pitch.angle, yaw_pitch.speed/100)
        yaw_pitch.last_angle = yaw_pitch.angle
        yaw_pitch.angle = false
    }
}

function updateJoint(motion, joint, pitch_roll) {
    if (pitch_roll.angle !== pitch_roll.last_angle) { // update pitch
        motion.setAngles(joint, pitch_roll.angle, pitch_roll.speed/100)
        pitch_roll.last_angle = pitch_roll.angle
        delete pitch_roll.angle
        delete pitch_roll.speed
    }
}

function updateJoints(motion) {
    const joints = [ 'shoulder', 'elbow', 'wrist' ]
    const capitalize = (str) => str.charAt(0).toUpperCase() + str.slice(1);

    Object.keys(state).forEach(joint => {
        if (joints.includes(joint)) {
            ['left', 'right'].forEach(position => {
                ['yaw', 'pitch', 'roll'].forEach(direction => {
                    let roll_pitch = state[joint][position][direction]
                    if (roll_pitch && roll_pitch.hasOwnProperty('angle')) {
                        const name = capitalize(position.charAt(0)) + capitalize(joint) + capitalize(direction)
                        updateJoint(motion, name, roll_pitch)
                    }
                })
            })
        }
    })
}

function updateMovement(motion) {
    motion.moveIsActive().then((active) => {
        let state_motion = state.motion
        if (state_motion.sequence.length) {
            if (active && state_motion.interrupt) {
                motion.stopMove()
            }
            if (!active) {
                (state_motion.sequence.shift())()
            }
        }
    })
}

function updateAudioOutput(ap, tts) {
    if (audioSequence.length) {
        const speechNotActive = ["stopped", "done"].includes(robot.TextToSpeech.Status)
        const audioFileActive = robot.AudioPlayer.playing
        if (audioInterrupt) {
            ap.stopAll()
            tts.stopAll()

            robot.TextToSpeech.Status = "stopped" // TODO: not sure if necesarry
            robot.AudioPlayer.playing = false
        }

        if (speechNotActive && !audioFileActive) {
            audioSequence[0]()
            audioSequence.shift()
        }
    }
}

function updateRobot() {
    if (session) {
        session.service("ALMotion").then((motion) => {
        updateYawPitch(motion, 'HeadYaw', state.head.yaw)
        updateYawPitch(motion, 'HeadPitch', state.head.pitch)
        updateHand(motion, 'LHand', state.hand.left)
        updateHand(motion, 'RHand', state.hand.right)

        updateJoints(motion)
        updateMovement(motion)

        session.service("ALAudioPlayer").then(ap => {
            session.service("ALTextToSpeech").then(tts => {
                updateAudioOutput(ap, tts)
                setTimeout(updateRobot, 1000/25) // i.e. x times per second
            }).fail(log_error)
        }).fail(log_error)
        }).fail(log_error)
    } else {
        setTimeout(updateRobot, 1000) // i.e. try again in a second...
    }
}

const DIRECTION_STEPS = {
    'forward': {x:1, y:0},
    'back': {x:-1, y:0},
    'left': {x:0, y:1, mirror:true},
    'right': {x:0, y:-1, mirror:true}
}

self.walk = (direction, steps, interrupt) => {
    let state_motion = state.motion
    if (interrupt) {
        state_motion.sequence = []
    }
    state_motion.interrupt = interrupt
    let length = steps * STEP_LENGTH
    let {x, y, mirror} = DIRECTION_STEPS[direction]
    x = x * length
    y = y * length
    if (state.mirror && mirror) {
        y = -y
    }
    state_motion.sequence.push(() => {
        session.service("ALMotion").then((mProxy) => {
            mProxy.moveTo(x, y, 0)
        })
    })
}

self.rotateBody = (angle, left, interrupt = false) => { //angle in degrees
    let direction = -1
    if (left) { direction = 1 }
    angle = helper_ConvertAngleToRads(angle)
    let state_motion = state.motion
    if (interrupt) {
        state_motion.sequence = []
    }
    state_motion.interrupt = interrupt
    if (state.mirror) {
        direction *= -1
    }
    state_motion.sequence.push(() => {
        session.service("ALMotion").then((mProxy) => {
            mProxy.moveTo(0, 0, angle * direction)
        })
    })
}

self.changeHand = (left, open) => {
    if (state.mirror) {
        left = !left
    }
    if (left) {
        state.hand.left.open = open
    } else { // right
        state.hand.right.open = open
    }
}

function updateHand(motion, hand, left_right) {
    if ((typeof left_right.open === "boolean")
        && (left_right.open !=  left_right.last_open)) { // update
        motion.killTasksUsingResources([hand]) // Kill any current motions
        if (left_right.open) {
            motion.openHand(hand)
        } else {
            motion.closeHand(hand)
        }
        left_right.last_open = left_right.open
        left_right.open = null // clear it...
    }
}

self.personPerception = (callback) => {
    session.service("ALBasicAwareness").then((ba) => {
        ba.startAwareness()
        _start_perception(session, callback)
    }).fail(log_error)
    destructor.add(() => {
        _destroy_perception(session)
    })
}

self.changeAutonomousLife = (state, callback) => {
    session.service("ALAutonomousLife").then((al) => {
        setTimeout(() => { // setTimeout override set_up behaviour
            al.setState(state).then(() => {
                if (callback) callback()
            })
        }, 100)
    }).fail(log_error)
}

self.createWordList = function (listName) {
    var v = new vocabList(listName, [])
    self._list.push(v)
}

self.addToWordList = function (listName, vocab) {
    if (!self._list.some(list => list.listName == listName)) self.createWordList(listName)
    
    for (var i = 0; i < self._list.length; i++) {
        if (self._list[i].listName == listName) {
            self._list[i].vocab = self._list[i].vocab.concat(vocab)
        }
    }
}


self.listenForWords = function (listName, vocab, confidence, blockID, callback, destruct = true) {
    if (!listName.length) { listName = "default" }

    self.addToWordList(listName, vocab)

    var list;
    var fullList = [];
    for (var i = 0; i < self._list.length; i++) {
        var element = self._list[i];
        self._list[i].vocab.forEach(function (word) {
            if (word.length) {
                fullList.push(word);
            }
        });
        if (element.listName == listName) {
            list = element;
        }
    }
    console.log(fullList);
    self.listen(session, list, fullList, confidence, blockID, callback, destruct);
}

self.changePosture = (pose, speed) => {
    if (speed <= 0) {
        speed = 1
    } else {
        speed = speed/100
    }
    session.service("ALRobotPosture").then((posture) => {
        posture.goToPosture(pose, speed)
    }).fail(log_error)
}

self.stopListening = function (callback) {
    session.service("ALSpeechRecognition").then(function (sr) {
        sr.unsubscribe("NAO_USER");
    });
}


/**
 * Class instance to store event disconnector
 * @param {function} disconnect Stores the function used to sidconnect the event
 */
var disEvent = class {
    constructor(disconnect) { this.disconnect = disconnect }
}

/** Stores the disEvents  */
let speechRecognitionEvents = []
let disconnectSpeechRecognition = false
/**
 * Starts the listener for the robot listening for words, also creates
 * the disconnect event
 * @param {object} session Stores the connection to the robot
 * @param {strings} list Stores the list of words to respond to
 * @param {number} confidence Stores the confidence value
 * @param {function} callback Function to execute if a word from list is heard
 * @param {object} sr Stores the speech recognition
 */
function _start_word_recognition(session, list, confidence, callback, sr) {
    session.service("ALMemory").then(function (ALMemory) {
        ALMemory.subscriber("WordRecognized").then(function (sub) {
            sub.signal.connect(function (value) {
                console.log(value)
                if (value[1] >= confidence) {
                    if (list.length) {
                        for (var i = 0; i < list.vocab.length; i++) {
                            if (callback && list.vocab[i] == value[0]) {
                                callback(value[0])
                            }
                        }
                    } else {
                        callback(value[0])
                    }
                }
            }).then(function (processID) {
                disconnectSpeechRecognition = () => {
                    sub.signal.disconnect(processID)
                }
                SRE = new disEvent(disconnectSpeechRecognition)
                speechRecognitionEvents.push(SRE)
            })
        })
    })
}

/**
 * Stops the robot listening 
 * @param {object} session Stores the connection to the robot
 * @param {string} blockID ID of block associated with the current listen event
 */
function _destroy_robot_listen(session, blockID) {
    session.service("ALSpeechRecognition").then(function (sr) {
        sr.unsubscribe(blockID)
    }).fail(log_error)
    speechRecognitionEvents.forEach(function (SRE) {
        SRE.disconnect()
    })
}

/** Stores the person perception event distructors */
let perceptionEvents = []
let disconnectPerception = false
/**
 * Makes the robot execute the function if he sees someone
 * Also creates the disconnect event
 * @param {object} session Stores the connection to the robot
 * @param {function} callback Function to execute if the robot sees someone
 */
function _start_perception(session, callback) {
    session.service("ALMemory").then(function (ALMemory) {
        ALMemory.subscriber("ALBasicAwareness/HumanTracked").then(function (sub) {
            sub.signal.connect(function (state) {
                if (state != -1) callback()
            }).then(function (processID) {
                disconnectPerception = () => {
                    sub.signal.disconnect(processID)
                }
                PE = new disEvent(disconnectPerception)
                perceptionEvents.push(PE)
            })
        })
    })
}

/**
 * Destructor for the person perception
 */
function _destroy_perception() {
    perceptionEvents.forEach(function (PE) {
        PE.disconnect()
    })
    session.service("ALBasicAwareness").then(function (ba) {
        ba.stopAwareness()
    }).fail(log_error)
}

/**
 * 
 * @param {object} session Stores the connection to the robot
 * @param {strings} list Stores the words to listen for 
 * @param {strings} fullList Stores all the lists of strings
 * @param {number} confidence Stores the confidence value for the robot to use
 * @param {string} blockID Stores the blockID for disconnection later
 * @param {function} callback Stores the function to execute
 * @param {boolean} destruct If it needs a destructor or not
 */
self.listen = function (session, list, fullList, confidence, blockID, callback, destruct = true) {
    session.service("ALSpeechRecognition").then(function (sr) {
        sr.setVocabulary(fullList, false)
        sr.setAudioExpression(false)
        sr.subscribe(blockID)

        _start_word_recognition(session, list, confidence, callback, sr)

    }).fail(log_error)
    if (destruct) {
        destructor.add(function () {
            _destroy_robot_listen(session, blockID) //create the destructor
        })
    }
}

/** Stores the disconnect events for the touchEvents */
let touchEvents = []
let disconnectTouch = false
/**
 * Sets up the listener for the touch events
 * Also sets up the disconnector
 * @param {object} session Stores the connection to the robot
 * @param {string} sensor Name of the sensor
 * @param {function} callback Function to execute
 */
function _start_touchEvents(session, sensor, callback) {
    session.service("ALMemory").then((ALMemory) => {
        ALMemory.subscriber(sensor).then((sub) => {
            // console.log(sub.signal)
            sub.signal.connect((state) => {
                if (state == 1) {
                    callback()
                }
            }).then((processID) => {
                disconnectTouch = () => {
                    sub.signal.disconnect(processID)
                }
                let touchEvent = new disEvent(disconnectTouch)
                touchEvents.push(touchEvent)
            })
        })
    })
}

/**
 * Destroys the touch events by executing the disconnectors
 */
function _destroy_touchEvents() {
    touchEvents.forEach((touchEvent) => {
        touchEvent.disconnect()
    })
    touchEvents = []
}

/**
 * Starts the touch events listener
 * @param {string} sensor Stores the sensor to listen to
 * @param {function} callback Stores the function to execute
 */
self.touchHead = (location, callback) => {
    _start_touchEvents(session, location + 'TactilTouched', callback)
    destructor.add(_destroy_touchEvents)
}

self.touchHand = (left, location, callback) => {
    if (state.mirror) {
        left = !left
    }
    let sensor = 'Hand' + (left?'Left':'Right')
    if (location == 'front') {
        sensor += left?'Right':'Left'
    } else if (location == 'back') {
        sensor += left?'Left':'Right'
    } else { // must be 'plate'
        sensor += 'Back'
    }
    sensor += 'Touched'
    _start_touchEvents(session, sensor, callback)
    destructor.add(_destroy_touchEvents)
}

self.touchFoot = (location, callback) => {
    let left = ['Left', 'Either'].includes(location)
    let right = ['Right', 'Either'].includes(location)
    if (state.mirror) {
        [left, right] = [right, left]
    }
    const suffix = 'BumperPressed'
    if (left) {
        _start_touchEvents(session, 'Left' + suffix, callback)
    }
    if (right) {
        _start_touchEvents(session, 'Right' + suffix, callback)
    }
    destructor.add(_destroy_touchEvents)
}

setTimeout(updateRobot, 1000) // Start in a second
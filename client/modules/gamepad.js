const quando = window['quando']
if (!quando) {
  alert('Fatal Error: gamepad.js must be included after quando_browser')
}

const FLIP = -1, UP = 0, DOWN = 1
const DOWN_UP = {'down': DOWN, 'up': UP, 'flip': FLIP}

let self = quando.gamepad = {}
let button_handlers = {}
let axis_handlers = {}
let last_buttons = []
let last_axes = []

function _handle(lookup, id, down_up, callback) {
  let handlers= lookup[id]
  if (handlers === undefined) {
    handlers = []
  }
  handlers.push({'down_up':down_up, 'callback':callback})
  lookup[id] = handlers
}

self.handleButton = (button_id, down_up, callback) => {
    let down_up_code = DOWN_UP[down_up]
    _handle(button_handlers, button_id, down_up_code, callback)
}

window.addEventListener("gamepadconnected", (event) => {
    let gamepad_in_use = _getGamepad()
    if (gamepad_in_use) {
        if (event.gamepad.index == gamepad_in_use.index) { // i.e. this is the gamepad to be used
            console.log("Using Gamepad " + gamepad_in_use.id)
            let buttons = gamepad_in_use.buttons
            for (let i=0; i<buttons.length; i++) {
                last_buttons[i] = 0 // set unpressed initially
            }
            let axes = gamepad_in_use.axes
            for (let i=0; i<axes.length; i++) {
                last_axes[i] = 0.5 // set in middle initially
            }
        }
    }
})

window.addEventListener("gamepaddisconnected", (event) => {
  last_button = []
  last_axes = []
  console.log("...Gamepad disconnected")
})

function _getGamepad() {
    let list = navigator.getGamepads()
    // return first valid gamepad found
    for (let gamepad of list) {
        if (gamepad) {
            return gamepad
        }
    }
    return false
}

function _updateButtons() {
  let gamepad = _getGamepad()
  if (gamepad) {
    let buttons = gamepad.buttons
    for (let i=0; i<buttons.length; i++) {
        // Check for change in value
        let value = buttons[i].value
        if (value != last_buttons[i]) {
            last_buttons[i] = value // update value
                if (button_handlers[i] !== undefined) {
                for (let handler of button_handlers[i]) {
                    let callback = handler['callback']
                    let down_up = handler['down_up']
                    if (value > 0) { // now pressed
                        if ((down_up == DOWN) || (down_up == FLIP)) {
                            callback(value)
                        }
                    } else { // now released
                        if ((down_up == UP) || (down_up == FLIP)) {
                            callback(value)
                        }
                    }
                }
            }
        }
    }
  }
}

// N.B. The triggers are added as axes, but actually held as button handlers
const AXIS_MAP = {true:{'x':0, 'y':1,'trigger':6}, false:{'x':2, 'y':3,'trigger':7}}

self.handleAxis = (left, axis, middle, plus_minus, ignore, inverted, callback) => {
  middle /= 100
  plus_minus /= 100
  ignore /= 100
  let half_ignore = ignore / 2
  let min_max_scaler = quando.new_scaler(middle-plus_minus, middle+plus_minus, inverted)
  let id = AXIS_MAP[left][axis]
  let ignore_handler = (val) => {
    let new_val = min_max_scaler(val)
    // Adjust for deadzone if not at limit
    if ((new_val > 0) || (new_val < 1)) {
      let fraction_from_middle = Math.abs(0.5 - new_val) *2 // i.e. /0.5
      if (fraction_from_middle <= ignore) { // or could compare against half_ignore
        new_val = 0.5
      } else { // scale the remaining fraction
        if (new_val < 0.5) { // i.e. below middle
            new_val /= 1-ignore
        } else { // above middle
            let above = new_val - 0.5 - half_ignore
            above /= 1-ignore // scale
            new_val = above + 0.5
        }
      }
    }
    return callback(new_val)
  }
  _handle(axis_handlers, id, FLIP, ignore_handler) 
}

function _updateAxes() {
  let gamepad = _getGamepad()
  if (gamepad) {
    let axes = gamepad.axes
    for (let i=0; i<axes.length; i++) {
        // Check for change in value
        let value = axes[i]
        if (value != last_axes[i]) {
            last_axes[i] = value // update value
            // Now convert to 0 to 1
            value = (value+1)/2
            // if x axis, invert
            if ((i % 2) == 1) {
                value = 1 - value
            }
            if (axis_handlers[i] !== undefined) {
                for (let handler of axis_handlers[i]) {
                    let callback = handler['callback']
                    callback(value)
                }
            }
        }
    }
  }
}

function _update() {
    _updateButtons()
    _updateAxes()
}

setInterval(_update, 1/30 * 1000) // 30 times a second
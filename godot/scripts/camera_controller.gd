extends Camera3D

## Orbital camera controller.
## Mouse drag to orbit, scroll to zoom, WASD to pan, Q/E to orbit horizontally.

@export var orbit_speed: float = 0.005
@export var zoom_speed: float = 500.0
@export var pan_speed: float = 200.0
@export var min_distance: float = 100.0
@export var max_distance: float = 50000.0

var _distance: float = 15000.0
var _yaw: float = 0.0       # horizontal angle (rad)
var _pitch: float = -0.6    # vertical angle (rad), negative = looking down
var _focus: Vector3 = Vector3.ZERO
var _dragging: bool = false


func _ready() -> void:
	_update_transform()


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventMouseButton:
		var mb := event as InputEventMouseButton
		if mb.button_index == MOUSE_BUTTON_RIGHT:
			_dragging = mb.pressed
		elif mb.button_index == MOUSE_BUTTON_WHEEL_UP:
			_distance = max(_distance - zoom_speed, min_distance)
			_update_transform()
		elif mb.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			_distance = min(_distance + zoom_speed, max_distance)
			_update_transform()
	elif event is InputEventMouseMotion and _dragging:
		var mm := event as InputEventMouseMotion
		_yaw -= mm.relative.x * orbit_speed
		_pitch = clamp(_pitch - mm.relative.y * orbit_speed, -1.5, -0.05)
		_update_transform()


func _process(delta: float) -> void:
	var pan := Vector3.ZERO
	if Input.is_key_pressed(KEY_W):
		pan.z -= 1.0
	if Input.is_key_pressed(KEY_S):
		pan.z += 1.0
	if Input.is_key_pressed(KEY_A):
		pan.x -= 1.0
	if Input.is_key_pressed(KEY_D):
		pan.x += 1.0

	if Input.is_key_pressed(KEY_Q):
		_yaw += 1.5 * delta
	if Input.is_key_pressed(KEY_E):
		_yaw -= 1.5 * delta

	if pan.length() > 0.0:
		# Pan relative to camera's horizontal orientation
		var forward := Vector3(-sin(_yaw), 0.0, -cos(_yaw))
		var right := Vector3(cos(_yaw), 0.0, -sin(_yaw))
		_focus += (forward * pan.z + right * pan.x) * pan_speed * delta
		_update_transform()
	elif Input.is_key_pressed(KEY_Q) or Input.is_key_pressed(KEY_E):
		_update_transform()


func set_focus(pos: Vector3) -> void:
	_focus = pos
	_update_transform()


func _update_transform() -> void:
	var offset := Vector3(
		_distance * cos(_pitch) * sin(_yaw),
		-_distance * sin(_pitch),
		_distance * cos(_pitch) * cos(_yaw),
	)
	global_position = _focus + offset
	look_at(_focus, Vector3.UP)

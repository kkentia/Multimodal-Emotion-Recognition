extends CharacterBody3D

# More reasonable values
const SPEED = 200
const SPRINT_SPEED = 400
const JUMP_VELOCITY = 200
const MOUSE_SENSITIVITY = 0.003
const DEATH_HEIGHT = -500.0
const ISLAND_RADIUS = 250.0

@export var spawn_position := Vector3(0, 100, 0)
var gravity = 500

@onready var camera = $Camera3D
@export var terrain: Node3D

# Spawn protection
var terrain_ready = false
var spawn_timer = 0.0
const MAX_WAIT = 3.0   # max seconds to wait for terrain

func _ready():
	Input.set_mouse_mode(Input.MOUSE_MODE_CAPTURED)
	global_position = spawn_position

func _input(event):
	# Only rotate camera when mouse is captured (no popup open)
	if event is InputEventMouseMotion and Input.mouse_mode == Input.MOUSE_MODE_CAPTURED:
		rotate_y(-event.relative.x * MOUSE_SENSITIVITY)
		camera.rotate_x(-event.relative.y * MOUSE_SENSITIVITY)
		camera.rotation.x = clamp(camera.rotation.x, -PI/2, PI/2)
	
	if event.is_action_pressed("ui_cancel"):
		Input.set_mouse_mode(Input.MOUSE_MODE_VISIBLE)

func _physics_process(delta):

	
	# === NORMAL PLAY ===
	var distance_from_center = Vector2(global_position.x, global_position.z).length()
	var on_island = distance_from_center < ISLAND_RADIUS
	
	var terrain_height = 0.0
	var has_ground = false
	if on_island and terrain and terrain.has_method("get_terrain_height"):
		terrain_height = terrain.get_terrain_height(global_position.x, global_position.z)
		has_ground = true
	
	var ground_y = terrain_height + 0.9
	var on_ground = has_ground and global_position.y <= ground_y + 0.1
	
	# Vertical
	if on_ground and velocity.y <= 0:
		velocity.y = 0
		global_position.y = ground_y
		if Input.is_action_just_pressed("ui_accept"):
			velocity.y = JUMP_VELOCITY
	else:
		var gravity_multiplier = 10 if velocity.y < 0 else 1.0
		velocity.y -= gravity * gravity_multiplier * delta
	
	# Horizontal
	var current_speed = SPRINT_SPEED if Input.is_key_pressed(KEY_SHIFT) else SPEED
	var input_dir = Input.get_vector("move_left", "move_right", "move_forward", "move_back")
	var direction = (transform.basis * Vector3(input_dir.x, 0, input_dir.y)).normalized()
	
	if direction:
		velocity.x = direction.x * current_speed
		velocity.z = direction.z * current_speed
	else:
		velocity.x = move_toward(velocity.x, 0, current_speed)
		velocity.z = move_toward(velocity.z, 0, current_speed)
	
	move_and_slide()
	
	# Death check
	if global_position.y < DEATH_HEIGHT:
		die_and_respawn()


func die_and_respawn():
	print("💀 You fell into the void!")
	global_position = spawn_position
	velocity = Vector3.ZERO
	# Trigger spawn grace period again so terrain reloads under us
	terrain_ready = false
	spawn_timer = 0.0

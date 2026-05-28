extends CharacterBody3D

const SPEED = 200
const SPRINT_SPEED = 400
const JUMP_VELOCITY = 200
const MOUSE_SENSITIVITY = 0.003
const GRAVITY_STRENGTH = 500
const DEATH_HEIGHT = -1000.0

# === LAUNCH MECHANICS ===
const LAUNCH_FORCE = 1500.0           # how fast we shoot toward target planet
const LAUNCH_RAY_LENGTH = 5000.0      # how far the aim ray reaches
const LAUNCH_DETECTION_RADIUS = 200.0 # extra "magnetism" radius around planets

@export var spawn_position := Vector3(0, 220, 0)
@export var planets: Array[Node3D] = []   # drag all planet nodes here

@onready var camera = $Camera3D
@onready var jump_sound = $JumpSound


var current_up := Vector3.UP
var current_planet: Node3D = null
var is_launching: bool = false           # while flying between planets, gravity is overridden

func _ready():
	Input.set_mouse_mode(Input.MOUSE_MODE_CAPTURED)
	global_position = spawn_position
	
	# Auto-discover planets if not assigned
	if planets.is_empty():
		for p in get_tree().get_nodes_in_group("planets"):
			planets.append(p)

func _input(event):
	if event is InputEventMouseMotion and Input.mouse_mode == Input.MOUSE_MODE_CAPTURED:
		rotate(current_up, -event.relative.x * MOUSE_SENSITIVITY)
		camera.rotate_x(-event.relative.y * MOUSE_SENSITIVITY)
		camera.rotation.x = clamp(camera.rotation.x, -PI/2, PI/2)

func _physics_process(delta):
	# === Find the nearest planet (this is "current_planet") ===
	current_planet = find_nearest_planet()
	if not current_planet:
		return
	
	var planet_center = current_planet.global_position
	var to_player = global_position - planet_center
	current_up = to_player.normalized()
	up_direction = current_up
	var gravity_dir = -current_up
	
	# === Reorient player ===
	align_with_gravity(delta)
	
	# === LAUNCH: jump toward planet you're aiming at ===
	if Input.is_action_just_pressed("ui_accept"):
		var target_planet = get_aimed_planet()
		if target_planet and target_planet != current_planet:
			launch_toward(target_planet)
			jump_sound.play()
		elif is_on_planet_surface():
			# Normal jump
			velocity += current_up * JUMP_VELOCITY
			
	
	# === Apply gravity toward nearest planet ===
	velocity += gravity_dir * GRAVITY_STRENGTH * delta
	
	# === Movement (only when on/near surface, not while launching) ===
	if is_near_surface():
		is_launching = false   # stop launch mode when we land
		apply_movement_input()
	# while launching: keep momentum, don't override velocity
	
	move_and_slide()


# === Find the nearest planet to apply gravity ===
func find_nearest_planet() -> Node3D:
	var nearest: Node3D = null
	var min_dist: float = INF
	for p in planets:
		if p == null:
			continue
		var d = global_position.distance_to(p.global_position)
		if d < min_dist:
			min_dist = d
			nearest = p
	return nearest


# === Detect which planet we're aiming at via raycast ===
func get_aimed_planet() -> Node3D:
	var space_state = get_world_3d().direct_space_state
	var from = camera.global_position
	var to = from + (-camera.global_transform.basis.z * LAUNCH_RAY_LENGTH)
	
	var query = PhysicsRayQueryParameters3D.create(from, to)
	query.exclude = [self]
	var result = space_state.intersect_ray(query)
	
	if result and result.has("collider"):
		var hit = result.collider
		# Check if the hit object is a planet (or child of one)
		for p in planets:
			if hit == p or hit.is_ancestor_of(p) or p.is_ancestor_of(hit):
				return p
	return null


# === Launch player toward target planet ===
func launch_toward(target: Node3D):
	print("🚀 Launching toward ", target.name)
	is_launching = true
	var direction = (target.global_position - global_position).normalized()
	velocity = direction * LAUNCH_FORCE


# === Helpers for surface detection ===
func is_on_planet_surface() -> bool:
	var radius = get_planet_radius(current_planet)
	var dist = global_position.distance_to(current_planet.global_position)
	return dist <= radius + 3.0

func is_near_surface() -> bool:
	var radius = get_planet_radius(current_planet)
	var dist = global_position.distance_to(current_planet.global_position)
	return dist <= radius + 30.0

func get_planet_radius(planet: Node3D) -> float:
	# Try to grab it from the CollisionShape3D
	var col = planet.find_child("CollisionShape3D", true, false)
	if col and col.shape is SphereShape3D:
		return col.shape.radius
	return 200.0   # fallback


# === Apply WASD movement (only on surface) ===
func apply_movement_input():
	var input_dir = Input.get_vector("move_left", "move_right", "move_forward", "move_back")
	var current_speed = SPRINT_SPEED if Input.is_key_pressed(KEY_SHIFT) else SPEED
	
	var forward = -transform.basis.z
	var right = transform.basis.x
	forward = (forward - forward.dot(current_up) * current_up).normalized()
	right = (right - right.dot(current_up) * current_up).normalized()
	
	var move_dir = (right * input_dir.x - forward * input_dir.y).normalized()
	
	var vertical_velocity = velocity.dot(current_up) * current_up
	var horizontal_velocity = move_dir * current_speed
	velocity = horizontal_velocity + vertical_velocity


# === Smoothly align player feet to planet surface ===
func align_with_gravity(delta):
	var current_basis = global_transform.basis
	var target_up = current_up
	var current_up_vec = current_basis.y
	var rotation_axis = current_up_vec.cross(target_up)
	
	if rotation_axis.length() > 0.001:
		var angle = current_up_vec.angle_to(target_up)
		rotation_axis = rotation_axis.normalized()
		var smooth_angle = angle * min(delta * 3.0, 1.0)
		global_transform.basis = current_basis.rotated(rotation_axis, smooth_angle)
		global_transform.basis = global_transform.basis.orthonormalized()

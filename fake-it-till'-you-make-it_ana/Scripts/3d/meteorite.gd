extends CharacterBody3D

var target_planet: Node3D = null
var fly_speed: float = 100.0   # Fast enough to be a threat
var max_lifetime: float = 60.0 # Despawn if it misses and flies forever

func _ready():
	# Point visually toward the target
	if target_planet:
		var dir = (target_planet.global_position - global_position).normalized()
		if dir.abs() != Vector3.UP:
			look_at(target_planet.global_position, Vector3.UP)

func _physics_process(delta):
	if not target_planet:
		return
		
	# Fly directly toward the target planet
	var direction = (target_planet.global_position - global_position).normalized()
	velocity = direction * fly_speed
	move_and_slide()
	
	# Spin wildly like an asteroid!
	$MeshInstance3D.rotate_object_local(Vector3(1, 0.5, 0).normalized(), 3.0 * delta)
	
	# Safety cleanup
	max_lifetime -= delta
	if max_lifetime <= 0:
		queue_free()

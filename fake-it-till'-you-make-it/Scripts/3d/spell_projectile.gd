extends Node3D

var velocity_dir: Vector3 = Vector3.FORWARD
var speed: float = 20.0
var damage: int = 10
var lifetime: float = 5.0

func _process(delta):
	global_position += velocity_dir * speed * delta
	
	# Spinning effect
	rotate_y(delta * 5.0)
	
	# Despawn after lifetime
	lifetime -= delta
	if lifetime <= 0:
		queue_free()


func _on_area_3d_body_entered(body):
	# Damage handler — when spell hits something with health
	if body.has_method("take_damage"):
		body.take_damage(damage)
	queue_free()

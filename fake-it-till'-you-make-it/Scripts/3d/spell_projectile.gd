extends Node3D

var velocity_dir: Vector3 = Vector3.FORWARD
var speed: float = 80.0
var damage: int = 10
var lifetime: float = 6.0
var launched: bool = false   # ← stays still until cast

func _process(delta):
	if not launched:
		return                # held in hand → don't move
	
	global_position += velocity_dir * speed * delta
	lifetime -= delta
	if lifetime <= 0:
		queue_free()

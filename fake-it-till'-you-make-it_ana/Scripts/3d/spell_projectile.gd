extends Node3D

var velocity_dir: Vector3 = Vector3.FORWARD
var speed: float = 80.0
var damage: int = 10
var lifetime: float = 6.0
var launched: bool = false
var drain_on_hit: bool = false

func _ready():
	$Area3D.body_entered.connect(_on_area_3d_body_entered)

func _process(delta):
	if not launched:
		return

	global_position += velocity_dir * speed * delta
	rotate_y(delta * 5.0)

	lifetime -= delta
	if lifetime <= 0:
		queue_free()

func _on_area_3d_body_entered(body):
	if body.has_method("take_damage"):
		body.take_damage(damage)
		if drain_on_hit:
			var root = get_tree().current_scene
			if root.has_method("heal_player"):
				root.heal_player(damage / 2)
	_hit_feedback()

func _hit_feedback():
	set_process(false)
	var tween = create_tween()
	tween.tween_property(self, "scale", scale * 2.0, 0.08)
	tween.tween_property(self, "scale", Vector3.ZERO, 0.08)
	tween.tween_callback(queue_free)

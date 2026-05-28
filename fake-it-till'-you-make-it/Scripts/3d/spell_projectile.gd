extends Node3D

var spell_name: String = ""
var display_name: String = ""
var spell_color: Color = Color.WHITE
var base_speed: float = 20.0
var velocity_dir: Vector3 = Vector3.FORWARD
var speed: float = 80.0
var damage: int = 10
var lifetime: float = 6.0
var launched: bool = false


func configure_projectile(
	p_spell_name: String,
	p_display_name: String,
	p_color: Color,
	p_base_speed: float,
	p_projectile_speed: float,
	p_damage: int,
	p_direction: Vector3
) -> void:
	spell_name = p_spell_name
	display_name = p_display_name
	spell_color = p_color
	base_speed = p_base_speed
	speed = p_projectile_speed
	damage = p_damage
	velocity_dir = p_direction.normalized()
	lifetime = 6.0
	launched = true
	print("🚀 Projectile configured: ", display_name, " dir=", velocity_dir, " speed=", speed)


func _process(delta: float) -> void:
	if not launched:
		return
	global_position += velocity_dir * speed * delta
	lifetime -= delta
	if lifetime <= 0:
		queue_free()


func _on_area_3d_body_entered(body: Node) -> void:
	if body.has_method("take_damage"):
		body.take_damage(damage)
	queue_free()

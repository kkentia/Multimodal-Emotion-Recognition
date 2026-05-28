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
var has_impacted: bool = false

@onready var area: Area3D = $Area3D
@onready var collision_shape: CollisionShape3D = $Area3D/CollisionShape3D


func _ready() -> void:
	if area != null:
		area.monitoring = true
		area.monitorable = true
		var body_callable: Callable = Callable(self, "_on_area_3d_body_entered")
		var area_callable: Callable = Callable(self, "_on_area_3d_area_entered")
		if not area.body_entered.is_connected(body_callable):
			area.body_entered.connect(body_callable)
		if not area.area_entered.is_connected(area_callable):
			area.area_entered.connect(area_callable)
	_expand_hitbox()


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


func _physics_process(delta: float) -> void:
	if not launched:
		return
	if _try_overlapping_targets():
		return
	var next_position: Vector3 = global_position + velocity_dir * speed * delta
	_try_hit_between(global_position, next_position)
	if has_impacted:
		return
	global_position = next_position
	lifetime -= delta
	if lifetime <= 0:
		queue_free()


func _on_area_3d_body_entered(body: Node) -> void:
	_try_damage_target(body)


func _on_area_3d_area_entered(hit_area: Area3D) -> void:
	_try_damage_target(hit_area)


func _expand_hitbox() -> void:
	if collision_shape == null or collision_shape.shape == null:
		return
	if collision_shape.shape is SphereShape3D:
		var sphere: SphereShape3D = collision_shape.shape as SphereShape3D
		sphere.radius = max(sphere.radius, 0.6)


func _try_hit_between(from_position: Vector3, to_position: Vector3) -> void:
	var space_state: PhysicsDirectSpaceState3D = get_world_3d().direct_space_state
	var query: PhysicsRayQueryParameters3D = PhysicsRayQueryParameters3D.create(from_position, to_position)
	query.collide_with_areas = true
	query.collide_with_bodies = true
	query.exclude = _get_initial_ray_excludes()

	for _attempt in range(12):
		var result: Dictionary = space_state.intersect_ray(query)
		if not result.has("collider"):
			return
		var collider: Node = result["collider"] as Node
		if _try_damage_target(collider):
			return
		if collider is CollisionObject3D:
			query.exclude.append((collider as CollisionObject3D).get_rid())
		else:
			return


func _try_overlapping_targets() -> bool:
	if area == null:
		return false
	for hit_area in area.get_overlapping_areas():
		if _try_damage_target(hit_area):
			return true
	for body in area.get_overlapping_bodies():
		if _try_damage_target(body):
			return true
	return false


func _try_damage_target(target: Node) -> bool:
	if has_impacted or target == null or not is_instance_valid(target):
		return false
	var damage_target: Node = target
	if not damage_target.has_method("take_damage"):
		damage_target = target.get_parent()
	if damage_target != null and damage_target.has_method("take_damage"):
		has_impacted = true
		damage_target.take_damage(damage)
		queue_free()
		return true
	return false


func _get_initial_ray_excludes() -> Array[RID]:
	var excludes: Array[RID] = []
	if area != null:
		excludes.append(area.get_rid())
	return excludes

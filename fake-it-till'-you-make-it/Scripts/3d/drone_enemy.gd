extends CharacterBody3D

signal destroyed(points: int)
signal attacked_player(damage: int)

@export var move_speed: float = 10.0
@export var static_target: bool = true
@export var max_health: int = 18
@export var hits_to_kill: int = 2
@export var point_value: int = 100
@export var contact_damage: int = 6
@export var attack_range: float = 24.0
@export var attack_cooldown: float = 1.4
@export var orbit_radius: float = 18.0
@export var orbit_pull_strength: float = 0.35
@export var orbit_spin_speed: float = 0.65
@export var vertical_follow_strength: float = 0.6
@export var bob_amplitude: float = 1.2

var current_health: int = 0
var hits_taken: int = 0
var hover_phase: float = 0.0
var last_attack_time: float = -10.0
var base_material: StandardMaterial3D
var orbit_direction: float = 1.0

@onready var mesh: MeshInstance3D = _find_primary_mesh()
@onready var collision_shape: CollisionShape3D = $CollisionShape3D
@onready var hitbox: Area3D = $Hitbox
@onready var hitbox_shape: CollisionShape3D = $Hitbox/CollisionShape3D
@onready var visual_root: Node3D = $Visual


func _ready() -> void:
	add_to_group("enemies")
	if hitbox != null:
		hitbox.add_to_group("enemy_hitboxes")
		hitbox.monitorable = true
		hitbox.monitoring = true
	current_health = hits_to_kill
	hover_phase = randf() * TAU
	orbit_direction = -1.0 if randf() < 0.5 else 1.0
	if mesh != null and mesh.get_active_material(0) is StandardMaterial3D:
		base_material = (mesh.get_active_material(0) as StandardMaterial3D).duplicate()
	elif mesh != null and mesh.material_override is StandardMaterial3D:
		base_material = (mesh.material_override as StandardMaterial3D).duplicate()
	else:
		base_material = StandardMaterial3D.new()
		base_material.albedo_color = Color(0.8, 0.2, 0.2, 1.0)
		base_material.emission_enabled = true
		base_material.emission = Color(0.8, 0.2, 0.2, 1.0)
		base_material.emission_energy_multiplier = 2.5
	if mesh != null:
		mesh.material_override = base_material


func _physics_process(delta: float) -> void:
	var current_scene: Node = _get_scene_root()
	if current_scene == null:
		return
	var target: Node3D = current_scene.get_node_or_null("Player")
	if target == null:
		return

	var target_up: Vector3 = Vector3.UP
	if "current_up" in target:
		target_up = target.current_up.normalized()

	if static_target:
		velocity = Vector3.ZERO
		if global_position.distance_to(target.global_position) > 0.1:
			look_at(target.global_position, target_up, true)
		return

	var to_target: Vector3 = target.global_position - global_position
	var vertical_error: float = to_target.dot(target_up)
	var horizontal_to_target: Vector3 = to_target - target_up * vertical_error
	var distance: float = horizontal_to_target.length()
	var radial_dir: Vector3 = horizontal_to_target.normalized() if distance > 0.01 else -target.global_transform.basis.z.normalized()
	var tangent_dir: Vector3 = target_up.cross(radial_dir).normalized() * orbit_direction
	var radial_offset := distance - orbit_radius
	var radial_velocity := radial_dir * radial_offset * move_speed * orbit_pull_strength
	var orbit_velocity := tangent_dir * move_speed * orbit_spin_speed
	var movement: Vector3 = radial_velocity + orbit_velocity
	movement += target_up * clamp(vertical_error, -12.0, 12.0) * vertical_follow_strength
	movement += target_up * sin((Time.get_ticks_msec() / 1000.0) + hover_phase) * bob_amplitude
	velocity = velocity.lerp(movement, min(delta * 3.5, 1.0))
	move_and_slide()
	if global_position.distance_to(target.global_position) > 0.1:
		look_at(target.global_position, target_up, true)

	if to_target.length() <= attack_range:
		_try_attack()


func take_damage(_amount: int) -> void:
	if current_health <= 0:
		return
	hits_taken += 1
	current_health = max(hits_to_kill - hits_taken, 0)
	_flash_hit()
	if hits_taken >= hits_to_kill:
		_die()


func _try_attack() -> void:
	var now: float = Time.get_ticks_msec() / 1000.0
	if (now - last_attack_time) < attack_cooldown:
		return
	last_attack_time = now
	attacked_player.emit(contact_damage)


func _flash_hit() -> void:
	if mesh == null or base_material == null:
		return
	if visual_root != null and is_instance_valid(visual_root):
		var hit_tween: Tween = create_tween()
		var enlarged_scale: Vector3 = visual_root.scale * 1.08
		hit_tween.tween_property(visual_root, "scale", enlarged_scale, 0.06)
		hit_tween.tween_property(visual_root, "scale", visual_root.scale, 0.1)
	base_material.albedo_color = Color(1.0, 0.35, 0.35, 1.0)
	base_material.emission = Color(1.0, 0.35, 0.35, 1.0)
	await get_tree().create_timer(0.12).timeout
	if is_instance_valid(mesh):
		base_material.albedo_color = Color(0.8, 0.2, 0.2, 1.0)
		base_material.emission = Color(0.8, 0.2, 0.2, 1.0)


func _find_primary_mesh() -> MeshInstance3D:
	var meshes: Array[Node] = find_children("*", "MeshInstance3D", true, false)
	if meshes.is_empty():
		return null
	return meshes[0] as MeshInstance3D


func _die() -> void:
	set_physics_process(false)
	velocity = Vector3.ZERO
	if collision_shape != null:
		collision_shape.set_deferred("disabled", true)
	if hitbox_shape != null:
		hitbox_shape.set_deferred("disabled", true)
	_spawn_death_burst()
	destroyed.emit(point_value)
	queue_free()


func _spawn_death_burst() -> void:
	var root: Node = _get_scene_root()
	if root == null:
		return

	var burst: Node3D = Node3D.new()
	root.add_child(burst)
	burst.global_position = global_position

	var flash_mesh: MeshInstance3D = MeshInstance3D.new()
	var sphere_mesh: SphereMesh = SphereMesh.new()
	sphere_mesh.radius = 0.6
	sphere_mesh.height = 1.2
	flash_mesh.mesh = sphere_mesh

	var flash_material: StandardMaterial3D = StandardMaterial3D.new()
	flash_material.albedo_color = Color(1.0, 0.45, 0.15, 0.9)
	flash_material.emission_enabled = true
	flash_material.emission = Color(1.0, 0.55, 0.2, 1.0)
	flash_material.emission_energy_multiplier = 5.0
	flash_material.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	flash_mesh.material_override = flash_material
	burst.add_child(flash_mesh)

	var flash_light: OmniLight3D = OmniLight3D.new()
	flash_light.light_color = Color(1.0, 0.5, 0.2, 1.0)
	flash_light.light_energy = 6.0
	flash_light.omni_range = 12.0
	burst.add_child(flash_light)

	var burst_tween: Tween = burst.create_tween()
	burst_tween.set_parallel(true)
	burst_tween.tween_property(flash_mesh, "scale", Vector3.ONE * 4.5, 0.22)
	burst_tween.tween_property(flash_light, "light_energy", 0.0, 0.22)
	burst_tween.chain().tween_callback(burst.queue_free)


func _get_scene_root() -> Node:
	if get_tree().current_scene != null:
		return get_tree().current_scene

	var node: Node = self
	while node.get_parent() != null and node.get_parent() != get_tree().root:
		node = node.get_parent()
	return node

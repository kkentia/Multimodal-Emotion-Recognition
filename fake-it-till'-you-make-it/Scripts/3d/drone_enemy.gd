extends CharacterBody3D

signal destroyed(points: int, killed_by_player:bool)
signal attacked_player(damage: int)   # kept so main_3d.gd connections don't break

@export var move_speed: float = 60.0
@export var hits_to_kill: int = 2
@export var point_value: int = 100
@export var planet_contact_damage: int = 1

var current_health: int = 0
var hits_taken: int = 0
var base_material: StandardMaterial3D

# --- TARGET A PLANET, NOT THE PLAYER ---
var target_planet: Node3D = null

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
	_setup_material()

	# If no target was assigned by spawner, grab a random planet ourselves
	if target_planet == null:
		var planets := get_tree().get_nodes_in_group("planets")
		if planets.is_empty():
			planets = get_tree().get_nodes_in_group("Planets")
		if not planets.is_empty():
			target_planet = planets.pick_random()
		else:
			push_warning("⚠️ Drone has no target planet and none found in groups")


func _physics_process(delta: float) -> void:
	if target_planet == null or not is_instance_valid(target_planet):
		queue_free()
		return

	var to_target: Vector3 = target_planet.global_position - global_position
	var distance: float = to_target.length()
	var direction: Vector3 = to_target.normalized() if distance > 0.01 else Vector3.ZERO

	velocity = direction * move_speed

	# Face the planet we're heading toward
	if distance > 0.1:
		look_at(target_planet.global_position, Vector3.UP)

	move_and_slide()

	# Hit the planet?
	for i in get_slide_collision_count():
			var collision := get_slide_collision(i)
			var collider := collision.get_collider()
			if collider != null and (collider.is_in_group("planets") or collider.is_in_group("Planets")):
				if collider.has_method("take_damage"):
					collider.take_damage(planet_contact_damage)
					print("☄️ Drone hit ", collider.name, "!")
				_die(false)   # ← crashed into planet, NOT a player kill
				return


func take_damage(_amount: int) -> void:
	if current_health <= 0:
		return
	hits_taken += 1
	current_health = max(hits_to_kill - hits_taken, 0)
	_flash_hit()
	if hits_taken >= hits_to_kill:
		_die(true)   # killed by player's spell


func _setup_material() -> void:
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


func _die(killed_by_player: bool = false) -> void:
	set_physics_process(false)
	velocity = Vector3.ZERO
	if collision_shape != null:
		collision_shape.set_deferred("disabled", true)
	if hitbox_shape != null:
		hitbox_shape.set_deferred("disabled", true)
	_spawn_death_burst()
	destroyed.emit(point_value, killed_by_player)   # ← pass the cause
	queue_free()


func _spawn_death_burst() -> void:
	var root: Node = get_tree().current_scene
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

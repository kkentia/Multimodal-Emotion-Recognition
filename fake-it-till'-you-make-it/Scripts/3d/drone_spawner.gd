extends Node3D

signal wave_started(wave_number: int)
signal enemy_spawned(enemy: Node)

@export var enemy_scene: PackedScene
@export var spawn_anchor_paths: Array[Node3D] = []
@export var enemies_per_wave: int = 3
@export var max_alive: int = 6
@export var wave_interval: float = 6.0
@export var spawn_radius: float = 180.0
@export var min_player_distance: float = 55.0
@export var spawn_height_min: float = 18.0
@export var spawn_height_max: float = 45.0
@export var wave_growth_interval: int = 3
@export var enemies_added_per_growth: int = 1
@export var max_alive_added_per_growth: int = 1
@export var static_enemy_count: int = 9
@export var static_surface_offset: float = 60.0
@export var static_min_spacing: float = 35.0

var current_wave: int = 0
var alive_enemies: Array[Node] = []
var wave_timer: Timer


func _ready() -> void:
	wave_timer = Timer.new()
	wave_timer.wait_time = wave_interval
	wave_timer.one_shot = false
	wave_timer.autostart = false
	wave_timer.timeout.connect(_on_wave_timer_timeout)
	add_child(wave_timer)


func start_waves() -> void:
	_spawn_wave()
	wave_timer.start()


func spawn_static_enemies(count: int = -1) -> void:
	_cleanup_alive()
	var target_count: int = static_enemy_count if count < 0 else count
	var anchors: Array[Node3D] = _get_spawn_anchors()
	if anchors.is_empty():
		return
	for index in range(target_count):
		_spawn_static_enemy(index, anchors)


func _on_wave_timer_timeout() -> void:
	_spawn_wave()


func _spawn_wave() -> void:
	_cleanup_alive()
	current_wave += 1
	wave_started.emit(current_wave)

	var growth_steps: int = int(floor(float(current_wave - 1) / float(max(wave_growth_interval, 1))))
	var scaled_enemies_per_wave: int = enemies_per_wave + growth_steps * enemies_added_per_growth
	var scaled_max_alive: int = max_alive + growth_steps * max_alive_added_per_growth
	var available_slots: int = max(0, scaled_max_alive - alive_enemies.size())
	var to_spawn: int = min(scaled_enemies_per_wave, available_slots)
	for _i in range(to_spawn):
		_spawn_enemy()


func _spawn_enemy() -> void:
	if enemy_scene == null:
		return

	var enemy: Node3D = enemy_scene.instantiate()
	var parent_scene: Node = get_parent()
	parent_scene.add_child(enemy)
	enemy.global_position = _pick_spawn_position()
	alive_enemies.append(enemy)
	enemy_spawned.emit(enemy)


func _spawn_static_enemy(index: int, anchors: Array[Node3D]) -> void:
	if enemy_scene == null:
		return

	var enemy: Node3D = enemy_scene.instantiate()
	var parent_scene: Node = get_parent()
	parent_scene.add_child(enemy)
	var anchor: Node3D = anchors[index % anchors.size()]
	enemy.global_position = _pick_planet_surface_position(anchor)
	alive_enemies.append(enemy)
	enemy_spawned.emit(enemy)


func unregister_enemy(enemy: Node) -> void:
	alive_enemies.erase(enemy)


func _cleanup_alive() -> void:
	var remaining: Array[Node] = []
	for enemy in alive_enemies:
		if is_instance_valid(enemy):
			remaining.append(enemy)
	alive_enemies = remaining


func _pick_spawn_position() -> Vector3:
	var player: Node3D = _get_player()
	if player != null:
		return _pick_player_relative_spawn(player)

	var anchors: Array[Node3D] = _get_spawn_anchors()

	var center: Vector3 = global_position
	if not anchors.is_empty():
		center = anchors[randi() % anchors.size()].global_position

	var candidate: Vector3 = center
	for _attempt in range(10):
		var direction: Vector3 = Vector3(
			randf_range(-1.0, 1.0),
			randf_range(-0.3, 0.3),
			randf_range(-1.0, 1.0)
		).normalized()
		candidate = center + direction * spawn_radius
		if _is_safe_spawn_position(candidate):
			return candidate

	return candidate


func _pick_planet_surface_position(anchor: Node3D) -> Vector3:
	var center: Vector3 = anchor.global_position
	var radius: float = _get_anchor_radius(anchor)
	var candidate: Vector3 = center
	for _attempt in range(24):
		var direction: Vector3 = _random_unit_vector()
		candidate = center + direction * (radius + static_surface_offset)
		if _is_safe_spawn_position(candidate) and _is_far_from_static_enemies(candidate):
			return candidate
	return candidate


func _random_unit_vector() -> Vector3:
	var direction: Vector3 = Vector3.ZERO
	while direction.length_squared() < 0.01:
		direction = Vector3(
			randf_range(-1.0, 1.0),
			randf_range(-1.0, 1.0),
			randf_range(-1.0, 1.0)
		)
	return direction.normalized()


func _get_anchor_radius(anchor: Node3D) -> float:
	var collision_shape: CollisionShape3D = anchor.get_node_or_null("CollisionShape3D") as CollisionShape3D
	if collision_shape != null and collision_shape.shape is SphereShape3D:
		return (collision_shape.shape as SphereShape3D).radius

	var meshes: Array[Node] = anchor.find_children("*", "MeshInstance3D", true, false)
	for node in meshes:
		var mesh_instance: MeshInstance3D = node as MeshInstance3D
		if mesh_instance != null and mesh_instance.mesh is SphereMesh:
			return (mesh_instance.mesh as SphereMesh).radius

	return spawn_radius


func _get_spawn_anchors() -> Array[Node3D]:
	var anchors: Array[Node3D] = []
	for anchor in spawn_anchor_paths:
		if anchor != null and is_instance_valid(anchor):
			anchors.append(anchor)
	anchors.shuffle()
	return anchors


func _is_far_from_static_enemies(candidate: Vector3) -> bool:
	for enemy in alive_enemies:
		if is_instance_valid(enemy) and enemy is Node3D:
			if (enemy as Node3D).global_position.distance_to(candidate) < static_min_spacing:
				return false
	return true


func _pick_player_relative_spawn(player: Node3D) -> Vector3:
	var player_up: Vector3 = Vector3.UP
	if "current_up" in player:
		player_up = player.current_up.normalized()

	var forward: Vector3 = -player.global_transform.basis.z
	forward = (forward - player_up * forward.dot(player_up)).normalized()
	if forward.length() < 0.01:
		forward = player_up.cross(Vector3.RIGHT).normalized()
	var right: Vector3 = forward.cross(player_up).normalized()

	var candidate: Vector3 = player.global_position
	for _attempt in range(12):
		var angle: float = randf_range(0.0, TAU)
		var tangent_dir: Vector3 = (forward * cos(angle) + right * sin(angle)).normalized()
		var distance: float = randf_range(min_player_distance, max(min_player_distance + 1.0, spawn_radius))
		var height: float = randf_range(spawn_height_min, spawn_height_max)
		candidate = player.global_position + tangent_dir * distance + player_up * height
		if _is_safe_spawn_position(candidate):
			return candidate

	return candidate


func _is_safe_spawn_position(candidate: Vector3) -> bool:
	var player: Node3D = _get_player()
	if player == null:
		return true
	return candidate.distance_to(player.global_position) >= min_player_distance


func _get_player() -> Node3D:
	var parent_scene: Node = get_parent()
	if parent_scene == null:
		return null
	return parent_scene.get_node_or_null("Player") as Node3D

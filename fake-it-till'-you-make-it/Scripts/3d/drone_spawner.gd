extends Node3D

signal wave_started(wave_number: int)
signal enemy_spawned(enemy: Node)

@export var enemy_scene: PackedScene
@export var enemies_per_wave: int = 3
@export var max_alive: int = 8
@export var wave_interval: float = 6.0
@export var spawn_radius: float = 4000.0      # how far out in space drones spawn
@export var wave_growth_interval: int = 3
@export var enemies_added_per_growth: int = 1

# Kept for compatibility with main_3d.gd / editor — unused now
@export var spawn_anchor_paths: Array[Node3D] = []

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


# main_3d.gd calls this on startup
func spawn_static_enemies(_count: int = -1) -> void:
	start_waves()


func start_waves() -> void:
	call_deferred("_spawn_wave")   # wait for scene to finish loading
	wave_timer.start()


func _on_wave_timer_timeout() -> void:
	_spawn_wave()


func _spawn_wave() -> void:
	_cleanup_alive()
	current_wave += 1
	wave_started.emit(current_wave)

	var growth_steps: int = int(floor(float(current_wave - 1) / float(max(wave_growth_interval, 1))))
	var scaled_per_wave: int = enemies_per_wave + growth_steps * enemies_added_per_growth
	var available_slots: int = max(0, max_alive - alive_enemies.size())
	var to_spawn: int = min(scaled_per_wave, available_slots)

	for _i in range(to_spawn):
		_spawn_drone()


func _spawn_drone() -> void:
	if enemy_scene == null:
		print("❌ No enemy_scene assigned to DroneSpawner!")
		return

	var planets := get_tree().get_nodes_in_group("planets")
	if planets.is_empty():
		planets = get_tree().get_nodes_in_group("Planets")
	if planets.is_empty():
		print("❌ No planets found in group 'planets'!")
		return

	var target: Node3D = planets.pick_random()

	# Random point on a large sphere around the world (above, below, any direction)
	var u: float = randf()
	var v: float = randf()
	var theta: float = 2.0 * PI * u
	var phi: float = acos(2.0 * v - 1.0)
	var offset := Vector3(
		spawn_radius * sin(phi) * cos(theta),
		spawn_radius * sin(phi) * sin(theta),
		spawn_radius * cos(phi)
	)
	# Spawn relative to the targeted planet so the drone always heads inward
	var spawn_pos: Vector3 = target.global_position + offset

	var enemy: Node3D = enemy_scene.instantiate()
	get_parent().add_child(enemy)
	enemy.global_position = spawn_pos

	if "target_planet" in enemy:
		enemy.target_planet = target

	alive_enemies.append(enemy)
	enemy_spawned.emit(enemy)
	print("☄️ Spawned drone at ", spawn_pos, " → targeting ", target.name)


func unregister_enemy(enemy: Node) -> void:
	alive_enemies.erase(enemy)


func _cleanup_alive() -> void:
	var remaining: Array[Node] = []
	for enemy in alive_enemies:
		if is_instance_valid(enemy):
			remaining.append(enemy)
	alive_enemies = remaining

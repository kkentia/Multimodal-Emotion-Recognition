extends Node3D

@export var meteorite_scene: PackedScene
@export var spawn_radius: float = 4000.0  # Spawns 4000m away in deep space
@export var spawn_interval: float = 5.0   # One new meteorite every 5 seconds

var planets: Array[Node] = []
var spawn_timer: float = 0.0

func _ready():
	# Auto-find all planets!
	planets = get_tree().get_nodes_in_group("Planets")
	if planets.is_empty():
				push_error("❌ Spawner found NO PLANETS! Add them to the 'planets' group.")

func _process(delta):
	if planets.is_empty() or not meteorite_scene:
		return
		
	spawn_timer -= delta
	if spawn_timer <= 0:
		spawn_meteorite()
		spawn_timer = spawn_interval

func spawn_meteorite():
	var target = planets.pick_random()
	if target == null: return
		
	# Pick a random point on a giant sphere around origin
	var u = randf()
	var v = randf()
	var theta = 2.0 * PI * u
	var phi = acos(2.0 * v - 1.0)
		
	var x = spawn_radius * sin(phi) * cos(theta)
	var y = spawn_radius * sin(phi) * sin(theta)
	var z = spawn_radius * cos(phi)
		
	var spawn_pos = Vector3(x, y, z)
	
	# Spawn it
	var meteorite = meteorite_scene.instantiate()
	get_tree().current_scene.add_child(meteorite)
	
	meteorite.global_position = spawn_pos
	meteorite.target_planet = target
	
	print("☄️ INCOMING! Meteorite spawned at ", spawn_pos.round(), " targeting ", target.name)

extends Area2D

var max_health: int=5
var current_health: int = 5

@onready var health_bar = $"../EnemyHealthBar"
@onready var hurt_sound = $"../AudioStreamPlayer2D"


signal enemy_died
signal enemy_attacked

func _ready():
	health_bar.max_value = max_health
	health_bar.value = current_health
	var timer = Timer.new()
	timer.wait_time = 3.0
	timer.autostart = true
	timer.timeout.connect(_on_attack_timer)
	add_child(timer)

func take_dmg():
	current_health -= 1
	health_bar.value = current_health
	hurt_sound.play()
	
	#add effects or sounds later here
	#$Sprite2D.modulate =Color(1,0,0) 
	#await get_tree().create_timer(0.2).timeout
	#$Sprite2D.modulate=Color(1,1,1) #normal color
	
	if current_health <= 0:
		die()
		
		
		
func die():
	enemy_died.emit()
	queue_free()

func _on_attack_timer():
	enemy_attacked.emit()

func _input_event(viewport, event, shape_idx):
	if event is InputEventMouseButton and event.pressed:
		take_dmg()
	

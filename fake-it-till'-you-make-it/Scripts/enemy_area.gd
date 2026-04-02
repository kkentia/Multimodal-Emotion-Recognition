extends Area2D

var max_health: int=5
var current_health: int = 5

@onready var health_bar = $"../EnemyHealthBar"


signal enemy_died # to broadcast to main for defeated_enemies

func _ready():
	health_bar.max_value =max_health
	health_bar.value = current_health

func take_dmg():
	current_health -= 1
	health_bar.value = current_health
	
	#add effects or sounds later here
	#$Sprite2D.modulate =Color(1,0,0) 
	#await get_tree().create_timer(0.2).timeout
	#$Sprite2D.modulate=Color(1,1,1) #normal color
	
	if current_health <= 0:
		die()
		
		
		
func die():
	print("Enemy Defeated")
	enemy_died.emit() #announce it to main to count nb of defeated enemies
	queue_free() #deletes the enemy from the gamee

#testing damage manually onclick
func _input_event(viewport, event, shape_idx):
	if event is InputEventMouseButton and event.pressed:
		take_dmg()
	

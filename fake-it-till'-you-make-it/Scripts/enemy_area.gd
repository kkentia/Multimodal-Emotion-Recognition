extends Area2D

var max_health: int=5
var current_health: int = 5

@onready var health_bar = $"../EnemyHealthBar"

func _ready():
	health_bar.max_value =max_health
	health_bar.value = current_health

func take_dmg():
	current_health -= 1
	health_bar.value = current_health
	
	#add effects or sounds later here
	$Sprite2D.modulate =Color(1,0,0) 
	await get_tree().create_timer(0.2).timeout
	$Sprite2D.modulate=Color(1,1,1) #normal color
	
	if current_health <= 0:
		die()
		
		
		
func die():
	print("Enemy Defeated")
	queue_free() #deletes the enemy from the gamee

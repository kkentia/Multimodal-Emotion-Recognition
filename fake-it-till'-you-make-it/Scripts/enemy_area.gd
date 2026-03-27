extends Area2D

var health: int=5

func take_dmg():
	health -=1
	#add effects or sounds later here
	if health <= 0:
		die()
		
func die():
	print("Enemy Defeated")
	queue_free() #deletes the enemy from the gamee

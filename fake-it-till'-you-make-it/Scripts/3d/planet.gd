extends StaticBody3D

func take_damage(amount: int) -> void:
	# forward the damage to main scene's total shared healthl
	var main = get_tree().current_scene
	if main and main.has_method("apply_planet_hit"):
		main.apply_planet_hit(amount, self)

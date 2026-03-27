extends Node2D

const CONFIDENCE_THRESHOLD = 0.7

@onready var ui = $UI
@onready var enemy = $Enemy

func _ready() -> void:
	pass

func process_ai_input(spoken_word: String, fer_emotion: String, fer_prob: float, ser_emotion: String, ser_prob: float):
	# send data to the UI scene so it can display the txt 
	ui.update_ai_text(spoken_word, fer_emotion, fer_prob, ser_emotion, ser_prob)
	
	# thresholds
	if fer_prob < CONFIDENCE_THRESHOLD or ser_prob < CONFIDENCE_THRESHOLD:
		print("Emotion not strong enough! Try harder!")
		return
		
	spoken_word = spoken_word.to_lower().strip_edges()
		
	#cast spells
	if spoken_word == "ignite" and fer_emotion == "Angry" and ser_emotion == "Angry":
		cast_spell("Fireball")
	elif spoken_word == "baffle" and fer_emotion == "Happy" and ser_emotion == "Angry":
		cast_spell("Confusion")
		
func cast_spell(spell_name: String):
	print("Successfully cast: ", spell_name)
	if is_instance_valid(enemy):
		enemy.take_dmg()

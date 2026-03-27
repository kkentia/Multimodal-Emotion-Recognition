extends Node2D

const CONFIDENCE_THRESHOLD = 0.7

@onready var ui = $UI
@onready var enemy = $Enemy
@onready var audio_stream = $AudioStreamPlayer2D


#player stats:
var player_max_health : int = 100
var player_current_health: int = 100

func _ready() -> void:
	ui.update_player_health(player_current_health)

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
	elif spoken_word == "restore" and fer_emotion == "Happy" and ser_emotion == "Happy":
		cast_spell("Healing")
		
func cast_spell(spell_name: String):
	print("Successfully cast: ", spell_name)
	
	if spell_name == "Fireball":
		if is_instance_valid(enemy):
			enemy.take_dmg()
			
	elif spell_name=="Healing":
		player_current_health+=20
		if player_current_health > player_max_health:
			player_current_health=player_current_health
		ui.update_player_health(player_current_health)
	


func _on_audio_stream_player_2d_finished() -> void:
	audio_stream.play() #does i loop (i hope)

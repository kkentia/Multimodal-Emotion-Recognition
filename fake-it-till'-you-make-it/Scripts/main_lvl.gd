extends Node2D

const CONFIDENCE_THRESHOLD = 0.0

@onready var ui = $UI
@onready var enemy = $Enemy
@onready var enemy2 = $Enemy2
@onready var audio_stream = $AudioStreamPlayer2D
@onready var you_won_label = $you_won
@onready var game_over_label = $game_over
@onready var spell_label = $spell_label

var active_enemy = null


#player stats:
var player_max_health : int = 100
var player_current_health: int = 100
var defeated_enemies  = 0

func _ready() -> void:
	you_won_label.visible = false
	game_over_label.visible = false
	spell_label.visible = false
	enemy2.visible = false
	ui.update_player_health(player_current_health)
	enemy.get_node("enemy_area").enemy_died.connect(_on_enemy_died)
	enemy.get_node("enemy_area").enemy_attacked.connect(_on_enemy_attacked)
	active_enemy = enemy

func process_ai_input(spoken_word: String, fer_emotion: String, fer_prob: float, ser_emotion: String, ser_prob: float):
	# send data to the UI scene so it can display the txt 
	ui.update_ai_text(spoken_word, fer_emotion, fer_prob, ser_emotion, ser_prob)
	
	# thresholds
	if fer_prob < CONFIDENCE_THRESHOLD or ser_prob < CONFIDENCE_THRESHOLD:
		print("Emotion not strong enough! Try harder!")
		return
		
	spoken_word = spoken_word.to_lower().strip_edges()
	fer_emotion = fer_emotion.to_lower()
	ser_emotion = ser_emotion.to_lower()

	#cast spells
	if spoken_word == "ignite" and fer_emotion == "angry" and ser_emotion == "angry":
		cast_spell("Fireball")
	elif spoken_word == "baffle" and fer_emotion == "happy" and ser_emotion == "angry":
		cast_spell("Confusion")
	elif spoken_word == "restore" and fer_emotion == "happy" and ser_emotion == "happy":
		cast_spell("Healing")
	elif spoken_word == "freeze" and fer_emotion == "sad" and ser_emotion == "fear":
		cast_spell("IceShard")
	elif spoken_word == "strike" and fer_emotion == "surprise" and ser_emotion == "angry":
		cast_spell("Lightning")
	elif spoken_word == "drain" and fer_emotion == "sad" and ser_emotion == "sad":
		cast_spell("ShadowDrain")
		
func show_spell_label(text: String):
	spell_label.text = text
	spell_label.visible = true
	await get_tree().create_timer(1.5).timeout
	spell_label.visible = false

func cast_spell(spell_name: String):
	print("Successfully cast: ", spell_name)

	if spell_name == "Fireball":
		show_spell_label("Fireball!")
		if is_instance_valid(active_enemy):
			active_enemy.get_node("enemy_area").take_dmg()

	elif spell_name == "Confusion":
		show_spell_label("Confusion!")
		if is_instance_valid(active_enemy):
			var area = active_enemy.get_node("enemy_area")
			area.set_process(false)
			await get_tree().create_timer(2.0).timeout
			if is_instance_valid(area):
				area.set_process(true)

	elif spell_name == "Healing":
		show_spell_label("Healing!")
		player_current_health += 20
		if player_current_health > player_max_health:
			player_current_health = player_max_health
		ui.update_player_health(player_current_health)

	elif spell_name == "IceShard":
		show_spell_label("Ice Shard!")
		if is_instance_valid(active_enemy):
			var area = active_enemy.get_node("enemy_area")
			area.take_dmg()
			area.set_process(false)
			await get_tree().create_timer(4.0).timeout
			if is_instance_valid(area):
				area.set_process(true)

	elif spell_name == "Lightning":
		show_spell_label("Lightning!")
		if is_instance_valid(active_enemy):
			var area = active_enemy.get_node("enemy_area")
			area.take_dmg()
			area.take_dmg()

	elif spell_name == "ShadowDrain":
		show_spell_label("Shadow Drain!")
		if is_instance_valid(active_enemy):
			active_enemy.get_node("enemy_area").take_dmg()
		player_current_health += 10
		if player_current_health > player_max_health:
			player_current_health = player_max_health
		ui.update_player_health(player_current_health)
	


func _on_audio_stream_player_2d_finished() -> void:
	audio_stream.play() #does i loop (i hope)



func _on_enemy_died():
	defeated_enemies += 1
	if defeated_enemies == 1:
		enemy2.visible = true
		enemy2.get_node("enemy_area").enemy_died.connect(_on_enemy_died)
		enemy2.get_node("enemy_area").enemy_attacked.connect(_on_enemy_attacked)
		active_enemy = enemy2
	win()

	
func _on_enemy_attacked():
	player_current_health -= 5
	if player_current_health < 0:
		player_current_health = 0
	ui.update_player_health(player_current_health)
	if player_current_health <= 0:
		game_over_label.visible = true

func win():
	if defeated_enemies == 2:
		you_won_label.visible = true

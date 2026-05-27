extends Node3D

const CONFIDENCE_THRESHOLD = 0.0   # raise to 0.5 later

@onready var ui = $UI
@onready var spell_manager = $Player/SpellManager

var player_max_health: int = 100
var player_current_health: int = 100


func _ready():
	print("✅ Main3d scene loaded")
	if not spell_manager:
		push_error("❌ SpellManager not found at $Player/SpellManager")
	if not ui:
		push_error("❌ UI not found at $UI")
	else:
		ui.update_player_health(player_current_health)


func heal_player(amount: int) -> void:
	player_current_health = min(player_current_health + amount, player_max_health)
	if ui:
		ui.update_player_health(player_current_health)


func take_damage(amount: int) -> void:
	player_current_health = max(player_current_health - amount, 0)
	if ui:
		ui.update_player_health(player_current_health)


func process_ai_input(spoken_word: String, fer_emotion: String, fer_prob: float, ser_emotion: String, ser_prob: float):
	print("📥 process_ai_input called: ", fer_emotion, "/", ser_emotion, " conf:", fer_prob, "/", ser_prob)
	
	# Update UI display
	ui.update_ai_text(spoken_word, fer_emotion, fer_prob, ser_emotion, ser_prob)
	
	if SignalBus.manual_mode:
		return
		
	# Normalize
	spoken_word = spoken_word.to_lower().strip_edges()
	fer_emotion = fer_emotion.to_lower()
	ser_emotion = ser_emotion.to_lower()
	
	# Safety check
	if not spell_manager:
		return
	
	# === LIVE SPELL HOLDING ===
	if fer_prob >= CONFIDENCE_THRESHOLD and ser_prob >= CONFIDENCE_THRESHOLD:
		spell_manager.update_held_spell(fer_emotion, ser_emotion)
	else:
		spell_manager.clear_held_spell()
	
	# === CONFIDENCE GATE FOR CASTING ===
	if fer_prob < CONFIDENCE_THRESHOLD or ser_prob < CONFIDENCE_THRESHOLD:
		return   # ← properly indented now ✅
	
	# === CAST SPELL when keyword spoken AND emotion combo matches ===
	if spoken_word == "ignite" and fer_emotion == "angry" and ser_emotion == "angry":
		spell_manager.cast_spell("ignite")
	elif spoken_word == "baffle" and fer_emotion == "happy" and ser_emotion == "angry":
		spell_manager.cast_spell("baffle")
	elif spoken_word == "restore" and fer_emotion == "happy" and ser_emotion == "happy":
		spell_manager.cast_spell("restore")
	elif spoken_word == "freeze" and fer_emotion == "sad" and ser_emotion == "fear":
		spell_manager.cast_spell("freeze")
	elif spoken_word == "strike" and fer_emotion == "fear" and ser_emotion == "angry":
		spell_manager.cast_spell("strike")
	elif spoken_word == "drain" and fer_emotion == "sad" and ser_emotion == "sad":
		spell_manager.cast_spell("drain")

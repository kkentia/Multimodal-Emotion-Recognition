extends Node3D

const CONFIDENCE_THRESHOLD: float = 0.0
const CAST_CHAIN_BONUS: int = 25

enum FiringMode {
	STT_ONLY,
	FER_SER_STT,
	FER_STT,
	SER_STT,
	SER_ONLY,
	MANUAL,
}

const FIRING_MODE_ORDER: Array[int] = [
	FiringMode.STT_ONLY,
	FiringMode.FER_SER_STT,
	FiringMode.FER_STT,
	FiringMode.SER_STT,
	FiringMode.SER_ONLY,
	FiringMode.MANUAL,
]

@onready var ui = $UI
@onready var player = $Player
@onready var spell_manager = $Player/SpellManager
@onready var drone_spawner = $DroneSpawner

var player_max_health: int = 100
var player_current_health: int = 100
var score: int = 0
var enemies_destroyed: int = 0
var current_wave: int = 0
var firing_mode: int = FiringMode.STT_ONLY
var last_fire_time: float = -10.0
var cast_chain: int = 0

var last_face_emotion: String = "unknown"
var last_face_conf: float = 0.0
var last_ser_emotion: String = "unknown"
var last_ser_conf: float = 0.0
var last_spoken_word: String = ""
var last_ready_ability: String = ""
var manual_spell_names: Array[String] = []
var manual_spell_index: int = 0
var is_game_over: bool = false


func _ready() -> void:
	print("✅ Main3d scene loaded")
	if not spell_manager:
		push_error("❌ SpellManager not found at $Player/SpellManager")
	if not ui:
		push_error("❌ UI not found at $UI")
	if not drone_spawner:
		push_error("❌ DroneSpawner not found at $DroneSpawner")

	_refresh_metrics()
	_refresh_readiness(false, false, false, false, "", "")
	manual_spell_names = spell_manager.get_spell_names()
	if manual_spell_names.is_empty():
		push_error("❌ No abilities found in SpellManager")
	elif ui and ui.has_method("update_spellbook"):
		ui.update_spellbook(spell_manager.get_spellbook_entries())

	if drone_spawner:
		drone_spawner.enemy_spawned.connect(_on_enemy_spawned)
		drone_spawner.wave_started.connect(_on_wave_started)
		drone_spawner.spawn_static_enemies()


func _unhandled_input(event: InputEvent) -> void:
	if event.is_action_pressed("toggle_fire_mode"):
		_cycle_firing_mode()
	elif event.is_action_pressed("cycle_ability") and firing_mode == FiringMode.MANUAL:
		_cycle_manual_ability()
	elif event.is_action_pressed("fire_debug") and firing_mode == FiringMode.MANUAL:
		_try_manual_fire()


func process_ai_payload(payload: Dictionary) -> void:
	last_face_emotion = str(payload.get("face_emotion", "unknown")).to_lower()
	last_face_conf = float(payload.get("face_confidence", 0.0))
	last_ser_emotion = str(payload.get("speech_emotion", "unknown")).to_lower()
	last_ser_conf = float(payload.get("speech_confidence", 0.0))
	last_spoken_word = str(payload.get("spoken_word", "")).to_lower().strip_edges()

	var fer_ok: bool = last_face_conf >= CONFIDENCE_THRESHOLD and last_face_emotion != "unknown"
	var ser_ok: bool = last_ser_conf >= CONFIDENCE_THRESHOLD and last_ser_emotion != "unknown"

	var ability_name: String = ""
	if firing_mode == FiringMode.MANUAL:
		ability_name = spell_manager.get_held_spell_name()
		last_ready_ability = ability_name
		_refresh_readiness(true, true, ability_name != "", ability_name != "", ability_name, "manual", spell_manager.get_expected_keyword(ability_name))
		return

	var casting_state: Dictionary = _resolve_casting_state(fer_ok, ser_ok)
	ability_name = str(casting_state.get("ability_name", ""))
	var stt_ok: bool = bool(casting_state.get("stt_ok", false))
	var ready_to_fire: bool = bool(casting_state.get("ready_to_fire", false))
	var expected_keyword: String = str(casting_state.get("expected_keyword", ""))

	if ability_name != "":
		spell_manager.select_spell(ability_name)
	else:
		spell_manager.clear_held_spell()

	last_ready_ability = ability_name
	_refresh_readiness(fer_ok, ser_ok, stt_ok, ready_to_fire, ability_name, last_spoken_word, expected_keyword)

	if ready_to_fire:
		_try_mode_fire(last_spoken_word, bool(casting_state.get("requires_keyword", true)))


func apply_player_damage(amount: int) -> void:
	player_current_health = max(player_current_health - amount, 0)
	cast_chain = 0
	_refresh_metrics()
	if ui and ui.has_method("show_status_message"):
		ui.show_status_message("Hull Hit -%d" % amount, Color(1.0, 0.3, 0.25, 1.0))
	if player_current_health == 0:
		_refresh_readiness(false, false, false, false, "", "Hull critical")


func apply_player_heal(amount: int) -> void:
	player_current_health = min(player_current_health + amount, player_max_health)
	_refresh_metrics()
	if ui and ui.has_method("show_status_message"):
		ui.show_status_message("Hull Restored +%d" % amount, Color(0.25, 1.0, 0.75, 1.0))


func _try_multimodal_fire(spoken_word: String) -> void:
	if spell_manager.cast_spell(spoken_word):
		_mark_fired(true)


func _try_mode_fire(spoken_word: String, requires_keyword: bool) -> void:
	var fired: bool = false
	if requires_keyword:
		fired = spell_manager.cast_spell(spoken_word)
	else:
		fired = spell_manager.cast_current_ability(true)

	if fired:
		_mark_fired(true)


func _try_manual_fire() -> void:
	if not spell_manager.has_held_spell():
		_arm_manual_spell()
	if spell_manager.cast_current_ability(true):
		_mark_fired(false)


func _mark_fired(is_spellbook_cast: bool) -> void:
	if is_spellbook_cast:
		cast_chain += 1
		score += CAST_CHAIN_BONUS * cast_chain
		if ui and ui.has_method("show_status_message"):
			ui.show_status_message("Spell Chain x%d +%d" % [cast_chain, CAST_CHAIN_BONUS * cast_chain], Color(0.6, 0.9, 1.0, 1.0))
	else:
		cast_chain = 0
	_refresh_metrics()


func _cycle_firing_mode() -> void:
	var current_index: int = FIRING_MODE_ORDER.find(firing_mode)
	if current_index == -1:
		current_index = 0
	_set_firing_mode(FIRING_MODE_ORDER[(current_index + 1) % FIRING_MODE_ORDER.size()])


func set_firing_mode_by_name(mode_name: String) -> void:
	match mode_name:
		"FER_SER_STT":
			_set_firing_mode(FiringMode.FER_SER_STT)
		"FER_STT":
			_set_firing_mode(FiringMode.FER_STT)
		"SER_STT":
			_set_firing_mode(FiringMode.SER_STT)
		"STT_ONLY":
			_set_firing_mode(FiringMode.STT_ONLY)
		"SER_ONLY":
			_set_firing_mode(FiringMode.SER_ONLY)
		"MANUAL":
			_set_firing_mode(FiringMode.MANUAL)


func _set_firing_mode(new_mode: int) -> void:
	firing_mode = new_mode
	if firing_mode == FiringMode.MANUAL:
		_arm_manual_spell()
	else:
		spell_manager.clear_held_spell()
	_refresh_metrics()
	if firing_mode == FiringMode.MANUAL:
		var manual_spell: String = spell_manager.get_held_spell_name()
		_refresh_readiness(true, true, manual_spell != "", manual_spell != "", manual_spell, "manual", spell_manager.get_expected_keyword(manual_spell))
	else:
		_refresh_readiness(
			last_face_conf >= CONFIDENCE_THRESHOLD and last_face_emotion != "unknown",
			last_ser_conf >= CONFIDENCE_THRESHOLD and last_ser_emotion != "unknown",
			false,
			false,
			last_ready_ability,
			last_spoken_word,
			spell_manager.get_expected_keyword(last_ready_ability)
		)


func _refresh_metrics() -> void:
	ui.update_metrics(
		player_current_health,
		player_max_health,
		score,
		current_wave,
		enemies_destroyed,
		_get_mode_name(),
		cast_chain
	)


func _refresh_readiness(fer_ok: bool, ser_ok: bool, stt_ok: bool, ready: bool, ability_name: String, spoken_word: String, expected_keyword: String = "") -> void:
	ui.update_readiness(fer_ok, ser_ok, stt_ok, ready, ability_name, spoken_word, expected_keyword, _get_mode_name())


func _get_mode_name() -> String:
	match firing_mode:
		FiringMode.FER_SER_STT:
			return "FER+SER+STT"
		FiringMode.FER_STT:
			return "FER+STT"
		FiringMode.SER_STT:
			return "SER+STT"
		FiringMode.STT_ONLY:
			return "STT Only"
		FiringMode.SER_ONLY:
			return "SER Only"
		FiringMode.MANUAL:
			return "Manual"
	return "FER+SER+STT"


func _arm_manual_spell() -> void:
	if manual_spell_names.is_empty():
		return
	manual_spell_index = clamp(manual_spell_index, 0, manual_spell_names.size() - 1)
	var spell_name: String = manual_spell_names[manual_spell_index]
	spell_manager.select_spell(spell_name)
	last_ready_ability = spell_name
	_refresh_readiness(true, true, true, true, spell_name, "manual", spell_manager.get_expected_keyword(spell_name))


func _cycle_manual_ability() -> void:
	if manual_spell_names.is_empty():
		return
	manual_spell_index = (manual_spell_index + 1) % manual_spell_names.size()
	_arm_manual_spell()


func _on_wave_started(wave_number: int) -> void:
	current_wave = wave_number
	_refresh_metrics()


func _on_enemy_spawned(enemy: Node) -> void:
	if enemy.has_signal("destroyed"):
		enemy.destroyed.connect(_on_enemy_destroyed)
	if enemy.has_signal("attacked_player"):
		enemy.attacked_player.connect(_on_enemy_attacked_player)


func _on_enemy_destroyed(points: int, killed_by_player: bool) -> void:
	if killed_by_player:
		enemies_destroyed += 1          # ← only count real kills
		score += points
		if ui and ui.has_method("show_status_message"):
			ui.show_status_message("+%d Enemy Banished" % points, Color(1.0, 0.8, 0.25, 1.0))
	else:
		score = max(score - 100, 0)
		if ui and ui.has_method("show_status_message"):
			ui.show_status_message("-100 Planet Breached!", Color(1.0, 0.3, 0.25, 1.0))
	_refresh_metrics()

func _on_enemy_attacked_player(damage: int) -> void:
	apply_player_damage(damage)


func _resolve_casting_state(fer_ok: bool, ser_ok: bool) -> Dictionary:
	var ability_name: String = ""
	var stt_ok: bool = false
	var requires_keyword: bool = true

	match firing_mode:
		FiringMode.FER_SER_STT:
			if fer_ok and ser_ok:
				ability_name = spell_manager.find_matching_spell(last_face_emotion, last_ser_emotion)
			stt_ok = _spoken_word_matches(ability_name)
			requires_keyword = true
		FiringMode.FER_STT:
			if fer_ok and last_spoken_word != "":
				ability_name = spell_manager.find_spell_by_constraints(last_face_emotion, "", last_spoken_word)
			elif fer_ok:
				ability_name = spell_manager.find_spell_by_constraints(last_face_emotion, "", "")
			stt_ok = ability_name != "" and _spoken_word_matches(ability_name)
			requires_keyword = true
		FiringMode.SER_STT:
			if ser_ok and last_spoken_word != "":
				ability_name = spell_manager.find_spell_by_constraints("", last_ser_emotion, last_spoken_word)
			elif ser_ok:
				ability_name = spell_manager.find_spell_by_constraints("", last_ser_emotion, "")
			stt_ok = ability_name != "" and _spoken_word_matches(ability_name)
			requires_keyword = true
		FiringMode.STT_ONLY:
			if last_spoken_word != "":
				ability_name = spell_manager.find_spell_by_constraints("", "", last_spoken_word)
			stt_ok = ability_name != ""
			requires_keyword = true
		FiringMode.SER_ONLY:
			if ser_ok:
				ability_name = spell_manager.find_spell_by_constraints("", last_ser_emotion, "")
			stt_ok = false
			requires_keyword = false

	var ready_to_fire: bool = ability_name != "" and (stt_ok or not requires_keyword)
	return {
		"ability_name": ability_name,
		"stt_ok": stt_ok,
		"ready_to_fire": ready_to_fire,
		"expected_keyword": spell_manager.get_expected_keyword(ability_name),
		"requires_keyword": requires_keyword,
	}


func _spoken_word_matches(ability_name: String) -> bool:
	return ability_name != "" and last_spoken_word != "" and last_spoken_word == spell_manager.get_expected_keyword(ability_name)


func apply_planet_hit(amount: int, planet: Node) -> void:
	player_current_health = max(player_current_health - amount, 0)
	cast_chain = 0
	_refresh_metrics()
	
	if ui and ui.has_method("show_status_message"):
		ui.show_status_message("Planet Hit! -%d" % amount, Color(1.0, 0.3, 0.25, 1.0))
	
	print("🪐 ", planet.name, " was hit! Health now: ", player_current_health)
	
	if player_current_health <= 0:
		_game_over()
		

func _game_over() -> void:
	if is_game_over:
		return
	is_game_over = true
	
	print("💀 GAME OVER — all health depleted")
	
	# Stop drones
	if drone_spawner and drone_spawner.has_method("stop_waves"):
		drone_spawner.stop_waves()
	
	# Build and show the game over panel
	_show_game_over_panel()
	
	# Pause AFTER building the panel
	get_tree().paused = true


func _show_game_over_panel() -> void:
	Input.set_mouse_mode(Input.MOUSE_MODE_VISIBLE)   # ← show cursor so buttons are clickable

	var overlay := CanvasLayer.new()
	overlay.name = "GameOverOverlay"
	overlay.layer = 100   # on top of everything
	overlay.process_mode = Node.PROCESS_MODE_ALWAYS   # works while paused
	add_child(overlay)
	
	# Dim background
	var bg := ColorRect.new()
	bg.color = Color(0, 0, 0, 0.75)
	bg.anchor_right = 1.0
	bg.anchor_bottom = 1.0
	overlay.add_child(bg)
	
	# Center container
	var center := CenterContainer.new()
	center.anchor_right = 1.0
	center.anchor_bottom = 1.0
	overlay.add_child(center)
	
	var vbox := VBoxContainer.new()
	vbox.add_theme_constant_override("separation", 24)
	center.add_child(vbox)
	
	# "GAME OVER" label
	var title := Label.new()
	title.text = "GAME OVER"
	title.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	title.add_theme_font_size_override("font_size", 72)
	title.add_theme_color_override("font_color", Color(1.0, 0.25, 0.25))
	vbox.add_child(title)
	
	# Score label
	var score_label := Label.new()
	score_label.text = "Final Score: %d" % score
	score_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	score_label.add_theme_font_size_override("font_size", 32)
	score_label.add_theme_color_override("font_color", Color(1.0, 0.9, 0.5))
	vbox.add_child(score_label)
	
	# Restart button
	var restart_btn := Button.new()
	restart_btn.text = "Restart"
	restart_btn.custom_minimum_size = Vector2(220, 60)
	restart_btn.add_theme_font_size_override("font_size", 28)
	restart_btn.process_mode = Node.PROCESS_MODE_ALWAYS   # clickable while paused
	restart_btn.pressed.connect(_on_restart_pressed)
	vbox.add_child(restart_btn)
	
	# quit to main menu button
	var quit_btn := Button.new()
	quit_btn.text = "Main Menu"
	quit_btn.custom_minimum_size = Vector2(220, 50)
	quit_btn.add_theme_font_size_override("font_size", 24)
	quit_btn.process_mode = Node.PROCESS_MODE_ALWAYS
	quit_btn.pressed.connect(_on_quit_to_menu_pressed)
	vbox.add_child(quit_btn)


func _on_restart_pressed() -> void:
	get_tree().paused = false        # unpause before reloading
	get_tree().reload_current_scene() # restart the level

func _on_quit_to_menu_pressed() -> void:
	get_tree().paused = false   #must unpause before changing screens
	get_tree().change_scene_to_file("res://Scences/StartScreen.tscn")

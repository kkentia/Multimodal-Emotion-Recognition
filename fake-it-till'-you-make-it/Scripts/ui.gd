extends CanvasLayer

@onready var data_box: VBoxContainer = $DataHUD/VBoxContainer
@onready var ability_label: Label = $DataHUD/VBoxContainer/NLP
@onready var fer_label: Label = $DataHUD/VBoxContainer/FER
@onready var ser_label: Label = $DataHUD/VBoxContainer/SER
@onready var webcam_feed: TextureRect = $WebcamFeed
@onready var settings: CanvasItem = $Settings/settings
@onready var player_health_bar: ProgressBar = $PlayerHealthBar
@onready var audio_stream: AudioStreamPlayer2D = $"../AudioStreamPlayer2D"
@onready var audio_check_button: CheckButton = $Settings/settings/TextureRect/AudioCheckButton
@onready var crosshair: TextureRect = $Crosshair
@onready var casted_label: Label = $CastedLabel
@onready var opened_book: TextureRect = $OpenedBook
@onready var tutorial: CanvasItem = $tutorial
@onready var spellbook_btn: TextureButton = $SpellbookBtn

const ABILITY_DISPLAY: Dictionary = {
	"Fireball": "Solar Flare",
	"Confusion": "Pulse Jammer",
	"Healing": "Repair Wave",
	"Ice Shard": "Cryo Burst",
	"Lightning": "Plasma Arc",
	"Shadow Drain": "Gravity Well",
}

var udp_data: PacketPeerUDP = PacketPeerUDP.new()
var udp_video: PacketPeerUDP = PacketPeerUDP.new()
var udp_data_bound: bool = false
var udp_video_bound: bool = false

var stt_label: Label
var mode_label: Label
var readiness_label: Label
var score_label: Label
var wave_label: Label
var kills_label: Label
var hint_label: Label
var health_label: Label
var fer_state_label: Label
var ser_state_label: Label
var stt_state_label: Label
var transcript_label: Label
var chain_label: Label

func _ready() -> void:
	_ensure_status_labels()
	player_health_bar.max_value = 100
	audio_check_button.button_pressed = true
	_set_default_hud_state()
	update_mouse_and_crosshair()

	var data_bind_result: int = udp_data.bind(4242)
	var video_bind_result: int = udp_video.bind(4243)
	udp_data_bound = data_bind_result == OK
	udp_video_bound = video_bind_result == OK
	if data_bind_result != OK:
		push_warning("Failed to bind UDP data socket on port 4242")
	if video_bind_result != OK:
		push_warning("Failed to bind UDP video socket on port 4243")
	print("Godot UDP Servers Started...")


func _process(_delta: float) -> void:
	if udp_video_bound and udp_video.get_available_packet_count() > 0:
		var packet: PackedByteArray = udp_video.get_packet()
		var img: Image = Image.new()
		var error: int = img.load_jpg_from_buffer(packet)
		if error == OK:
			webcam_feed.texture = ImageTexture.create_from_image(img)

	if udp_data_bound and udp_data.get_available_packet_count() > 0:
		var packet: PackedByteArray = udp_data.get_packet()
		var json_string: String = packet.get_string_from_utf8()
		var payload: Variant = JSON.parse_string(json_string)
		if payload is Dictionary:
			var payload_dict: Dictionary = payload
			var spoken_word: String = str(payload_dict.get("spoken_word", ""))
			var transcript_text: String = str(payload_dict.get("transcript", ""))
			update_ai_text(
				str(payload_dict.get("spell", "")),
				str(payload_dict.get("face_emotion", "unknown")),
				float(payload_dict.get("face_confidence", 0.0)),
				str(payload_dict.get("speech_emotion", "unknown")),
				float(payload_dict.get("speech_confidence", 0.0)),
				spoken_word,
				transcript_text
			)

			var parent_node: Node = get_parent()
			if parent_node != null and is_instance_valid(parent_node) and parent_node.has_method("process_ai_payload"):
				parent_node.process_ai_payload(payload_dict)
			elif parent_node != null and is_instance_valid(parent_node) and parent_node.has_method("process_ai_input"):
				parent_node.process_ai_input(
					spoken_word,
					str(payload_dict.get("face_emotion", "unknown")),
					float(payload_dict.get("face_confidence", 0.0)),
					str(payload_dict.get("speech_emotion", "unknown")),
					float(payload_dict.get("speech_confidence", 0.0))
				)

	if Input.is_action_just_pressed("ui_cancel"):
		if settings.visible:
			settings.visible = false
		elif opened_book.visible:
			opened_book.visible = false
			spellbook_btn.button_pressed = false
		elif tutorial.visible:
			tutorial.visible = false
		else:
			settings.visible = true
		update_mouse_and_crosshair()
	if Input.is_action_just_pressed("toggle_spellbook"):
		_toggle_spellbook_panel()


func update_ai_text(spell_name: String, fer_emo: String, fer_prob: float, ser_emo: String, ser_prob: float, spoken_word: String = "", transcript_text: String = "") -> void:
	var display_ability: String = format_ability_name(spell_name)
	if spell_name == "" or spell_name == "none":
		display_ability = get_ability_name_from_emotions(fer_emo, ser_emo)

	ability_label.text = "Ability: %s" % display_ability
	fer_label.text = "FER: %s %d%%" % [fer_emo.capitalize(), int(clamp(fer_prob, 0.0, 1.0) * 100.0)]
	ser_label.text = "SER: %s %d%%" % [ser_emo.capitalize(), int(clamp(ser_prob, 0.0, 1.0) * 100.0)]
	stt_label.text = "STT Cue: %s" % (spoken_word if spoken_word != "" else "waiting")
	transcript_label.text = "Transcript: %s" % (transcript_text if transcript_text != "" else "...")


func update_metrics(current_health: int, max_health: int, score: int, wave: int, kills: int, mode_name: String, cast_chain: int = 0) -> void:
	player_health_bar.max_value = max_health
	player_health_bar.value = current_health
	health_label.text = "Health: %d / %d" % [current_health, max_health]
	score_label.text = "Score: %d" % score
	wave_label.text = "Wave: %d" % wave
	kills_label.text = "Kills: %d" % kills
	mode_label.text = "Mode: %s | M mode | E spellbook" % mode_name
	chain_label.text = "Spell Chain: x%d" % cast_chain


func update_readiness(fer_ready: bool, ser_ready: bool, stt_ready: bool, ready_to_fire: bool, ability_name: String, spoken_word: String, expected_keyword: String = "", mode_name: String = "") -> void:
	var status_parts: Array[String] = [
		"FER %s" % _flag_text(fer_ready),
		"SER %s" % _flag_text(ser_ready),
		"STT %s" % _flag_text(stt_ready)
	]
	readiness_label.text = "Ready: %s" % " | ".join(status_parts)
	fer_state_label.text = _indicator_text("FER", fer_ready)
	ser_state_label.text = _indicator_text("SER", ser_ready)
	stt_state_label.text = _indicator_text("STT", stt_ready)

	var detail: String = "No spellbook match yet"
	if ability_name != "":
		detail = "%s armed" % format_ability_name(ability_name)
		if expected_keyword != "" and not ready_to_fire and mode_name != "Manual":
			detail = "%s armed | say '%s'" % [format_ability_name(ability_name), expected_keyword]
	elif spoken_word != "" and spoken_word != "manual":
		detail = "No spellbook match for '%s'" % spoken_word
	if spoken_word == "manual":
		detail = "%s armed | LMB fire | F cycle" % format_ability_name(ability_name)
	if ready_to_fire:
		if mode_name == "Manual":
			detail = "%s manual override active" % format_ability_name(ability_name)
		elif expected_keyword != "":
			detail = "%s ready via '%s'" % [format_ability_name(ability_name), expected_keyword]
		else:
			detail = "%s ready" % format_ability_name(ability_name)
	hint_label.text = detail


func show_casted_message(spell_name: String) -> void:
	casted_label.text = "%s ACTIVATED" % format_ability_name(spell_name).to_upper()
	casted_label.visible = true
	casted_label.scale = Vector2(0.5, 0.5)
	var tween: Tween = create_tween()
	tween.tween_property(casted_label, "scale", Vector2(1.2, 1.2), 0.2)
	tween.tween_property(casted_label, "scale", Vector2(1.0, 1.0), 0.1)
	await get_tree().create_timer(1.5).timeout
	var fade_tween: Tween = create_tween()
	fade_tween.tween_property(casted_label, "modulate:a", 0.0, 0.3)
	await fade_tween.finished
	casted_label.visible = false
	casted_label.modulate.a = 1.0


func show_status_message(text: String, color: Color = Color.WHITE) -> void:
	casted_label.text = text
	casted_label.modulate = color
	casted_label.visible = true
	casted_label.scale = Vector2(0.8, 0.8)
	var tween: Tween = create_tween()
	tween.tween_property(casted_label, "scale", Vector2(1.0, 1.0), 0.12)
	await get_tree().create_timer(0.9).timeout
	var fade_tween: Tween = create_tween()
	fade_tween.tween_property(casted_label, "modulate:a", 0.0, 0.2)
	await fade_tween.finished
	casted_label.visible = false
	casted_label.modulate = Color.WHITE


func update_spellbook(entries: Array[Dictionary]) -> void:
	var spellbook_text: String = ""
	for entry in entries:
		spellbook_text += "- %s: say '%s'  -> Face: %s | Voice: %s\n" % [
			str(entry.get("display_name", "")),
			str(entry.get("keyword", "")),
			str(entry.get("face", "")),
			str(entry.get("voice", "")),
		]
	var book_label: Label = opened_book.get_node_or_null("Label") as Label
	if book_label != null:
		book_label.text = spellbook_text.strip_edges()


func update_mouse_and_crosshair() -> void:
	var any_popup_open: bool = settings.visible or opened_book.visible or tutorial.visible
	if any_popup_open:
		Input.set_mouse_mode(Input.MOUSE_MODE_VISIBLE)
		crosshair.visible = false
	else:
		Input.set_mouse_mode(Input.MOUSE_MODE_CAPTURED)
		crosshair.visible = true


func format_ability_name(name: String) -> String:
	return str(ABILITY_DISPLAY.get(name, name))


func get_ability_name_from_emotions(fer: String, ser: String) -> String:
	fer = fer.to_lower()
	ser = ser.to_lower()

	if fer == "angry" and ser == "angry":
		return "Solar Flare"
	elif fer == "happy" and ser == "angry":
		return "Pulse Jammer"
	elif fer == "happy" and ser == "happy":
		return "Repair Wave"
	elif fer == "sad" and ser == "fear":
		return "Cryo Burst"
	elif fer == "fear" and ser == "angry":
		return "Plasma Arc"
	elif fer == "sad" and ser == "sad":
		return "Gravity Well"
	return "—"


func _ensure_status_labels() -> void:
	health_label = _ensure_label("Health")
	stt_label = _ensure_label("STT")
	transcript_label = _ensure_label("Transcript")
	mode_label = _ensure_label("Mode")
	readiness_label = _ensure_label("Readiness")
	score_label = _ensure_label("Score")
	wave_label = _ensure_label("Wave")
	kills_label = _ensure_label("Kills")
	hint_label = _ensure_label("Hint")
	fer_state_label = _ensure_label("FERState")
	ser_state_label = _ensure_label("SERState")
	stt_state_label = _ensure_label("STTState")
	chain_label = _ensure_label("SpellChain")


func _ensure_label(node_name: String) -> Label:
	var label: Label = data_box.get_node_or_null(node_name) as Label
	if label == null:
		label = Label.new()
		label.name = node_name
		data_box.add_child(label)
	return label


func _set_default_hud_state() -> void:
	update_ai_text("", "unknown", 0.0, "unknown", 0.0, "")
	update_metrics(100, 100, 0, 0, 0, "FER+SER+STT")
	update_readiness(false, false, false, false, "", "")
	settings.visible = false
	opened_book.visible = false
	spellbook_btn.button_pressed = false
	tutorial.visible = true


func _exit_tree() -> void:
	udp_data.close()
	udp_video.close()


func _flag_text(state: bool) -> String:
	return "OK" if state else "--"


func _indicator_text(label_name: String, active: bool) -> String:
	return "%s: %s" % [label_name, "ON" if active else "OFF"]


func _toggle_spellbook_panel() -> void:
	opened_book.visible = not opened_book.visible
	spellbook_btn.button_pressed = opened_book.visible
	if opened_book.visible:
		settings.visible = false
		tutorial.visible = false
	update_mouse_and_crosshair()


func _on_spellbook_btn_pressed() -> void:
	_toggle_spellbook_panel()


func _on_spellbook_btn_toggled(toggled_on: bool) -> void:
	opened_book.visible = toggled_on
	if opened_book.visible:
		settings.visible = false
		tutorial.visible = false
	update_mouse_and_crosshair()


func _on_open_tutorial_pressed() -> void:
	tutorial.visible = true
	opened_book.visible = false
	spellbook_btn.button_pressed = false
	settings.visible = false
	update_mouse_and_crosshair()


func _on_settings_pressed() -> void:
	settings.visible = not settings.visible
	if settings.visible:
		opened_book.visible = false
		spellbook_btn.button_pressed = false
		tutorial.visible = false
	update_mouse_and_crosshair()


func _on_close_btn_pressed() -> void:
	settings.visible = false
	tutorial.visible = false
	update_mouse_and_crosshair()


func _on_exit_game_pressed() -> void:
	get_tree().change_scene_to_file("res://Scences/StartScreen.tscn")


func _on_check_button_toggled(toggled_on: bool) -> void:
	audio_stream.volume_db = 0 if toggled_on else -100


func _select_firing_mode(mode_name: String) -> void:
	var parent_node: Node = get_parent()
	if parent_node != null and parent_node.has_method("set_firing_mode_by_name"):
		parent_node.set_firing_mode_by_name(mode_name)


func _on_mode_fer_ser_stt_pressed() -> void:
	_select_firing_mode("FER_SER_STT")


func _on_mode_fer_stt_pressed() -> void:
	_select_firing_mode("FER_STT")


func _on_mode_ser_stt_pressed() -> void:
	_select_firing_mode("SER_STT")


func _on_mode_stt_only_pressed() -> void:
	_select_firing_mode("STT_ONLY")


func _on_mode_ser_only_pressed() -> void:
	_select_firing_mode("SER_ONLY")


func _on_mode_manual_pressed() -> void:
	_select_firing_mode("MANUAL")

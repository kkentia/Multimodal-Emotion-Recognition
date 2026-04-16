extends CanvasLayer

@onready var text_label = $DataHUD/VBoxContainer/NLP
@onready var fer_label = $DataHUD/VBoxContainer/FER
@onready var ser_label = $DataHUD/VBoxContainer/SER
@onready var webcam_feed = $WebcamFeed
@onready var opened_book = $OpenedBook
@onready var settings = $Settings/settings
@onready var tutorial = $tutorial
@onready var player_health_bar =$PlayerHealthBar
@onready var audio_stream = $"../AudioStreamPlayer2D"
@onready var audio_check_button= $Settings/settings/TextureRect/AudioCheckButton

# UDP listeners
# := sets the variable type to whatever is on the other side of the equal sign
var udp_data := PacketPeerUDP.new()
var udp_video := PacketPeerUDP.new()

func _ready() -> void:
	opened_book.visible= false
	settings.visible=false
	tutorial.visible=true
	audio_check_button.button_pressed=true
	
	
	
	#listen to python script server (resnet_UI_godot_bridge.py)
	udp_data.bind(4242)
	udp_video.bind(4243)
	print("Godot UDP Servers Started...")
		
	#udp_video.bind(4243)#audio
	print("Godot UDP Servers Started...")



func _process(_delta: float) -> void:  # fix the warning too
	if udp_video.get_available_packet_count() > 0:
		var packet = udp_video.get_packet()
		print("Video packet received, size: ", packet.size())  # are packets arriving?
		var img = Image.new()
		var error = img.load_jpg_from_buffer(packet)
		print("Image load error code: ", error)  # is 0 (OK)?
		if error == OK:
			webcam_feed.texture = ImageTexture.create_from_image(img)
			print("Texture applied")  # is this reached?

	# 2. RECEIVE JSON DATA
	if udp_data.get_available_packet_count() > 0:
		var packet = udp_data.get_packet()
		var json_string = packet.get_string_from_utf8()
		var dict = JSON.parse_string(json_string)
		
		if dict:
			#send the received py data to the UI text update func
			update_ai_text(
				dict["spell"], 
				dict["face_emotion"], dict["face_confidence"], 
				dict["speech_emotion"], dict["speech_confidence"]
			)
			
			#send data up to Main Level to cast spell
			if get_parent().has_method("process_ai_input"):
				get_parent().process_ai_input(
					dict["spell"], 
					dict["face_emotion"], dict["face_confidence"], 
					dict["speech_emotion"], dict["speech_confidence"]
				)
	
	if Input.is_action_just_pressed("ui_cancel"): #this is escape by default for some reason
		settings.visible= not settings.visible
	
func update_ai_text(spell: String, fer_emo: String, fer_prob: float, ser_emo: String, ser_prob: float):
	text_label.text = "Spell: " + str(spell)
	fer_label.text = "Face: " + fer_emo.capitalize() + " " + str(int(fer_prob * 100)) + "%"
	ser_label.text = "Voice: " + ser_emo.capitalize() + " " + str(int(ser_prob * 100)) + "%"




func update_player_health(current_health:int):
	player_health_bar.value=current_health





func _on_spellbook_btn_pressed() -> void:
	opened_book.visible = not opened_book.visible

func _on_settings_pressed() -> void:
	settings.visible = not settings.visible
	print("Settings toggled!")


func _on_close_btn_pressed() -> void:
	settings.visible= false
	tutorial.visible=false


func _on_exit_game_pressed() -> void:
	get_tree().change_scene_to_file("res://Scences/StartScreen.tscn")


func _on_open_tutorial_pressed() -> void:
	tutorial.visible=true


func _on_check_button_toggled(toggled_on: bool) -> void:
	if toggled_on:
		audio_stream.volume_db= 0 #unmute
	else: audio_stream.volume_db=-100 ##not audible to human ear
	


	

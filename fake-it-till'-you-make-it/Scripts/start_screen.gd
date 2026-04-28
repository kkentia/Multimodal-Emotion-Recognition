extends Control
@onready var loading_label = $TextureRect/PlayBtn/loading

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	loading_label.visible=false
	


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(_delta: float) -> void:
	pass


func _on_play_btn_pressed():
	if !loading_label.visible:
		loading_label.visible=true
	get_tree().change_scene_to_file("res://Scences/3d/main3d.tscn")
	#get_tree
	#change_scene_to_file completly unloads the Start screen and loads lvl1

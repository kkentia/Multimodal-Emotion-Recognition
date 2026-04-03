extends Control


# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	pass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(_delta: float) -> void:
	pass


func _on_play_btn_pressed():
	get_tree().change_scene_to_file("res://Scences/MainLVL.tscn")
	#get_tree
	#change_scene_to_file completly unloads the Start screen and loads lvl1

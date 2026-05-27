extends Node

@export var spell_hand: Node3D
@export var camera: Camera3D

var SPELLS = {
	"Fireball": {
		"scene": preload("res://Scences/3d/spells/fireball.tscn"),
		"face": "Angry", "voice": "Angry",
		"keyword": "ignite",
		"speed": 80.0,
		"damage": 25
	},
	"Confusion": {
		"scene": preload("res://Scences/3d/spells/confusion.tscn"),
		"face": "Happy", "voice": "Angry",
		"keyword": "baffle",
		"speed": 50.0,
		"damage": 10
	},
	"Healing": {
		"scene": preload("res://Scences/3d/spells/healing.tscn"),
		"face": "Happy", "voice": "Happy",
		"keyword": "restore",
		"speed": 0.0,
		"damage": -20
	},
	"Ice Shard": {
		"scene": preload("res://Scences/3d/spells/ice_shard.tscn"),
		"face": "Sad", "voice": "Fear",
		"keyword": "freeze",
		"speed": 70.0,
		"damage": 20
	},
	"Lightning": {
		"scene": preload("res://Scences/3d/spells/lightning.tscn"),
		"face": "Fear", "voice": "Angry",
		"keyword": "strike",
		"speed": 150.0,
		"damage": 30
	},
	"Shadow Drain": {
		"scene": preload("res://Scences/3d/spells/shadow_drain.tscn"),
		"face": "Sad", "voice": "Sad",
		"keyword": "drain",
		"speed": 60.0,
		"damage": 15
	}
}

var held_spell_node: Node3D = null
var held_spell_name: String = ""


func update_held_spell(face: String, voice: String):
	var matching_spell = ""
	for spell_name in SPELLS.keys():
		var spell = SPELLS[spell_name]
		if spell.face.to_lower() == face.to_lower() and spell.voice.to_lower() == voice.to_lower():
			matching_spell = spell_name
			break
	
	if matching_spell != held_spell_name:
		clear_held_spell()
		if matching_spell != "":
			spawn_held_spell(matching_spell)


func spawn_held_spell(spell_name: String):
	var spell = SPELLS[spell_name]
	var instance = spell.scene.instantiate()
	spell_hand.add_child(instance)
	instance.position = Vector3.ZERO
	
	held_spell_node = instance
	held_spell_name = spell_name
	print("✨ Holding spell: ", spell_name)


func clear_held_spell():
	if held_spell_node:
		held_spell_node.queue_free()
	held_spell_node = null
	held_spell_name = ""


func cast_spell(spoken_keyword: String):
	print("🔥 cast_spell got keyword: '", spoken_keyword, "' | held: '", held_spell_name, "'")
	push_error("CAST_SPELL CALLED")
	
	if held_spell_name == "":
		return
	
	var spell = SPELLS[held_spell_name]
	if spoken_keyword.to_lower() != spell.keyword:
		return
	
	print("🎯 Casting ", held_spell_name, "!")
	
	# === SHOW "CASTED" MESSAGE ON UI ===
	var ui = get_tree().current_scene.get_node_or_null("UI")
	print("🔍 Looking for UI: ", ui)
	if ui:
		print("🔍 UI has show_casted_message? ", ui.has_method("show_casted_message"))
		if ui.has_method("show_casted_message"):
			ui.show_casted_message(held_spell_name)
			print("✅ Called show_casted_message")
	
	var spell_to_cast_name = held_spell_name
	
	# Healing — self-cast, just disappear
	if spell.speed == 0:
		clear_held_spell()
		return
	
	# === LAUNCH THE SPELL ===
	var spell_instance = held_spell_node
	held_spell_node = null
	held_spell_name = ""

	var cam = get_viewport().get_camera_3d()

	if not cam:
		print("no acive pov found")
		return
		
	# DEBUG: print the raw global transform
	print("CAM global_transform.basis.z = ", cam.global_transform.basis.z)
	print("CAM global rotation = ", cam.global_rotation)
	print("CAM is current? ", cam.current, "  name: ", cam.name)
	# Grab camera's CURRENT forward direction and position RIGHT NOW
	var cam_transform = cam.global_transform
	var shoot_dir = -cam_transform.basis.z      # camera forward
	var shoot_origin = cam_transform.origin     # camera position

# Reparent so it flies freely in space
	spell_instance.get_parent().remove_child(spell_instance)

	# SET DIRECTION FIRST (before add_child triggers _ready)
	spell_instance.velocity_dir = shoot_dir
	spell_instance.speed = spell.speed
	spell_instance.damage = spell.damage
	spell_instance.lifetime = 6.0
	spell_instance.launched=true

	# NOW add to scene (this fires _ready)
	get_tree().current_scene.add_child(spell_instance)
	spell_instance.global_position = shoot_origin + shoot_dir * 3.0

	print("🎯 Shooting toward ", shoot_dir)
